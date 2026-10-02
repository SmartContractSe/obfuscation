from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import torch
from torch import nn
from torch.nn import functional as F
from solidity_parser import parser
from slither.core.cfg.node import NodeType
from slither.slither import Slither

RAW_EDGE_TYPES = (
    "AST",
    "AST_REV",
    "NEXT",
    "PREV",
    "DATA",
    "DATA_REV",
    "CONTROL",
    "CONTROL_REV",
    "EFFECT",
    "EFFECT_REV",
)
RAW_EDGE_TYPE_TO_ID = {name: index for (index, name) in enumerate(RAW_EDGE_TYPES)}
SEMANTIC_IDENTIFIERS = frozenset(
    {
        "block",
        "call",
        "callcode",
        "delegatecall",
        "gas",
        "msg",
        "now",
        "number",
        "origin",
        "selfdestruct",
        "send",
        "sender",
        "timestamp",
        "transfer",
        "tx",
        "value",
        "require",
        "assert",
        "revert",
    }
)
SCALAR_FIELDS = frozenset(
    {"operator", "visibility", "stateMutability", "isConstructor", "kind"}
)
STRUCTURAL_FIELDS = frozenset(
    {
        "arguments",
        "base",
        "body",
        "condition",
        "conditionExpression",
        "expression",
        "FalseBody",
        "FalseExpression",
        "index",
        "initialValue",
        "initExpression",
        "left",
        "loopExpression",
        "names",
        "parameters",
        "returnParameters",
        "right",
        "statements",
        "subExpression",
        "subNodes",
        "TrueBody",
        "TrueExpression",
        "typeName",
        "variables",
    }
)
IGNORED_FIELDS = frozenset({"loc", "src", "range"})


def stable_token_id(token: str, vocab_size: int = 32768) -> int:
    if vocab_size < 3:
        raise ValueError("vocab_size must be at least 3")
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (vocab_size - 1) + 1


@dataclass
class SemanticGraph:
    address: str
    domain: str
    source_path: str
    label: tuple[int, int, int]
    tokens: list[str]
    raw_tokens: list[str]
    node_kinds: list[str]
    parent: list[int]
    node_unit: list[int]
    unit_names: list[str]
    edge_src: list[int]
    edge_dst: list[int]
    edge_type: list[int]
    token_id_values: list[int] = field(default_factory=list)
    split: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    _tensor_cache: dict[str, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        node_count = len(self.tokens)
        if (
            not len(self.raw_tokens)
            == len(self.node_kinds)
            == len(self.parent)
            == len(self.node_unit)
            == node_count
        ):
            raise ValueError("node fields must have equal lengths")
        if not len(self.edge_src) == len(self.edge_dst) == len(self.edge_type):
            raise ValueError("edge fields must have equal lengths")
        if any((index < 0 or index >= node_count for index in self.edge_src)):
            raise ValueError("edge source outside graph")
        if any((index < 0 or index >= node_count for index in self.edge_dst)):
            raise ValueError("edge destination outside graph")

    @property
    def num_nodes(self) -> int:
        return len(self.tokens)

    @property
    def num_edges(self) -> int:
        return len(self.edge_src)

    @property
    def num_units(self) -> int:
        return len(self.unit_names)

    def token_ids(self, vocab_size: int = 32768) -> torch.Tensor:
        if vocab_size != 32768:
            raise ValueError("packed graphs use the fixed 32768-bucket vocabulary")
        if not self.token_id_values:
            self.token_id_values = [stable_token_id(token) for token in self.tokens]
        return self._cached_tensor("token_ids", self.token_id_values, torch.long)

    def edge_index(self) -> torch.Tensor:
        cached = self._tensor_cache.get("edge_index")
        if cached is None:
            cached = torch.tensor([self.edge_src, self.edge_dst], dtype=torch.long)
            self._tensor_cache["edge_index"] = cached
        return cached

    def edge_types(self) -> torch.Tensor:
        return self._cached_tensor("edge_type", self.edge_type, torch.long)

    def node_units(self) -> torch.Tensor:
        return self._cached_tensor("node_unit", self.node_unit, torch.long)

    def _cached_tensor(
        self, key: str, values: list[int], dtype: torch.dtype
    ) -> torch.Tensor:
        cached = self._tensor_cache.get(key)
        if cached is None:
            cached = torch.tensor(values, dtype=dtype)
            self._tensor_cache[key] = cached
        return cached


class MutableSemanticGraph:
    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.raw_tokens: list[str] = []
        self.node_kinds: list[str] = []
        self.parent: list[int] = []
        self.node_unit: list[int] = []
        self.unit_names: list[str] = []
        self.current_unit = -1
        self.edge_src: list[int] = []
        self.edge_dst: list[int] = []
        self.edge_type: list[int] = []

    def add_node(self, token: str, raw_token: str, kind: str, parent: int) -> int:
        index = len(self.tokens)
        self.tokens.append(token)
        self.raw_tokens.append(raw_token)
        self.node_kinds.append(kind)
        self.parent.append(parent)
        self.node_unit.append(self.current_unit)
        return index

    def add_edge(self, source: int, target: int, relation: str) -> int:
        index = len(self.edge_src)
        self.edge_src.append(source)
        self.edge_dst.append(target)
        self.edge_type.append(RAW_EDGE_TYPE_TO_ID[relation])
        return index

    def add_bidirectional(self, source: int, target: int, relation: str) -> None:
        reverse = {
            "AST": "AST_REV",
            "NEXT": "PREV",
            "DATA": "DATA_REV",
            "CONTROL": "CONTROL_REV",
            "EFFECT": "EFFECT_REV",
        }[relation]
        self.add_edge(source, target, relation)
        self.add_edge(target, source, reverse)


@dataclass(frozen=True)
class GraphBuildConfig:
    include_reverse_edges: bool = True
    include_sequence_edges: bool = True
    include_control_edges: bool = True
    include_effect_edges: bool = True


class SoliditySemanticGraphBuilder:
    def __init__(self, config: GraphBuildConfig | None = None) -> None:
        self.config = config or GraphBuildConfig()

    def from_ast(
        self,
        ast: dict[str, Any],
        *,
        address: str,
        domain: str,
        source_path: str,
        label: tuple[int, int, int],
        split: str,
    ) -> SemanticGraph:
        normalization_stats: dict[str, int] = {}
        self._state_variables = self._collect_state_variables(ast)
        graph = MutableSemanticGraph()
        functions = self._collect_functions(ast)
        if not functions:
            functions = [ast]
        for unit_index, function in enumerate(functions):
            graph.current_unit = unit_index
            raw_name = str(function.get("name") or f"anonymous_{unit_index}")
            graph.unit_names.append(raw_name)
            root = graph.add_node("ENTRY", raw_name, "entry", -1)
            identity_nodes: list[tuple[str, int]] = []
            self._visit_dict(function, root, "root", graph, identity_nodes)
        result = SemanticGraph(
            address=address,
            domain=domain,
            source_path=source_path,
            label=label,
            tokens=graph.tokens,
            raw_tokens=graph.raw_tokens,
            node_kinds=graph.node_kinds,
            parent=graph.parent,
            node_unit=graph.node_unit,
            unit_names=graph.unit_names,
            edge_src=graph.edge_src,
            edge_dst=graph.edge_dst,
            edge_type=graph.edge_type,
            token_id_values=[stable_token_id(token) for token in graph.tokens],
            split=split,
            metadata={
                "extractor": "isg_ea_rgcn_semantic_graph_v1",
                "normalization": normalization_stats,
            },
        )
        return result

    def _collect_state_variables(self, value: Any) -> set[str]:
        result: set[str] = set()

        def visit(current: Any) -> None:
            if isinstance(current, dict):
                if current.get("type") == "VariableDeclaration" and current.get(
                    "isStateVar"
                ):
                    if current.get("name"):
                        result.add(str(current["name"]))
                for child in current.values():
                    visit(child)
            elif isinstance(current, list):
                for child in current:
                    visit(child)

        visit(value)
        return result

    def _visit_dict(
        self,
        value: dict[str, Any],
        parent: int,
        field_name: str,
        graph: MutableSemanticGraph,
        identity_nodes: list[tuple[str, int]],
    ) -> int:
        node_type = str(value.get("type", "Object"))
        current = graph.add_node(f"type:{node_type}", node_type, "ast_type", parent)
        self._connect(graph, parent, current, "AST")
        child_roots: list[int] = []
        roots_by_key: dict[str, list[int]] = {}
        for key, child in value.items():
            if key in IGNORED_FIELDS or key == "type" or child is None:
                continue
            if isinstance(child, dict):
                field_parent = self._field_node(key, current, graph)
                child_root = self._visit_dict(
                    child, field_parent, key, graph, identity_nodes
                )
                child_roots.append(child_root)
                roots_by_key.setdefault(key, []).append(child_root)
            elif isinstance(child, list):
                if not child:
                    continue
                field_parent = self._field_node(key, current, graph)
                list_roots: list[int] = []
                semantic_items: list[dict[str, Any]] = []
                for item in child:
                    if isinstance(item, dict):
                        semantic_items.append(item)
                        list_roots.append(
                            self._visit_dict(
                                item, field_parent, key, graph, identity_nodes
                            )
                        )
                    elif self._keep_scalar(key, item, node_type):
                        list_roots.append(
                            self._scalar_node(
                                key,
                                item,
                                node_type,
                                field_parent,
                                graph,
                                identity_nodes,
                            )
                        )
                self._sequence_edges(graph, list_roots)
                if key == "statements" and self.config.include_effect_edges:
                    self._effect_edges(graph, semantic_items, list_roots)
                child_roots.extend(list_roots)
                roots_by_key.setdefault(key, []).extend(list_roots)
            elif self._keep_scalar(key, child, node_type):
                scalar_root = self._scalar_node(
                    key, child, node_type, current, graph, identity_nodes
                )
                child_roots.append(scalar_root)
                roots_by_key.setdefault(key, []).append(scalar_root)
        self._sequence_edges(graph, child_roots)
        if self.config.include_control_edges:
            self._control_edges(graph, node_type, roots_by_key)
        return current

    def _field_node(self, key: str, parent: int, graph: MutableSemanticGraph) -> int:
        if key not in STRUCTURAL_FIELDS:
            return parent
        node = graph.add_node(f"field:{key}", key, "field", parent)
        self._connect(graph, parent, node, "AST")
        return node

    @staticmethod
    def _keep_scalar(key: str, value: Any, node_type: str) -> bool:
        if key in SCALAR_FIELDS:
            return True
        if key in {"name", "memberName", "value", "number"}:
            return True
        return node_type.endswith("Literal") and isinstance(value, (str, int, bool))

    def _scalar_node(
        self,
        key: str,
        value: Any,
        owner_type: str,
        parent: int,
        graph: MutableSemanticGraph,
        identity_nodes: list[tuple[str, int]],
    ) -> int:
        raw = str(value)
        kind = "attribute"
        if key == "operator":
            token = f"op:{raw}"
            kind = "operator"
        elif key == "memberName":
            lowered = raw.lower()
            token = (
                f"identifier:{lowered}"
                if lowered in SEMANTIC_IDENTIFIERS
                else "identifier:<user>"
            )
            kind = "identifier"
        elif key == "name":
            lowered = raw.lower()
            globally_semantic = {
                "abi",
                "assert",
                "block",
                "msg",
                "now",
                "require",
                "revert",
                "selfdestruct",
                "this",
                "tx",
            }
            token = (
                f"identifier:{lowered}"
                if owner_type == "Identifier" and lowered in globally_semantic
                else "identifier:<user>"
            )
            kind = "identifier"
        elif key in {"number", "value"} or owner_type.endswith("Literal"):
            token = self._literal_token(owner_type, value)
            kind = "literal"
        else:
            token = f"attr:{key}:{raw}"
        node = graph.add_node(token, raw, kind, parent)
        self._connect(graph, parent, node, "AST")
        if kind == "identifier" and key == "name" and (token == "identifier:<user>"):
            identity_nodes.append((f"{kind}:{raw}", node))
        return node

    @staticmethod
    def _literal_token(owner_type: str, value: Any) -> str:
        if owner_type == "BooleanLiteral" or isinstance(value, bool):
            return f"literal:bool:{str(value).lower()}"
        text = str(value).replace("_", "")
        try:
            number = int(text, 0)
        except ValueError:
            return f"literal:{type(value).__name__}"
        if number == 0:
            bucket = "zero"
        elif number == 1:
            bucket = "one"
        elif number < 256:
            bucket = "small"
        else:
            bucket = "large"
        return f"literal:int:{bucket}"

    def _connect(
        self, graph: MutableSemanticGraph, source: int, target: int, relation: str
    ) -> None:
        if self.config.include_reverse_edges:
            graph.add_bidirectional(source, target, relation)
        else:
            graph.add_edge(source, target, relation)

    def _sequence_edges(self, graph: MutableSemanticGraph, nodes: list[int]) -> None:
        if not self.config.include_sequence_edges:
            return
        for left, right in zip(nodes, nodes[1:]):
            self._connect(graph, left, right, "NEXT")

    def _control_edges(
        self,
        graph: MutableSemanticGraph,
        node_type: str,
        roots_by_key: dict[str, list[int]],
    ) -> None:
        if node_type not in {
            "IfStatement",
            "WhileStatement",
            "DoWhileStatement",
            "ForStatement",
            "Conditional",
        }:
            return
        conditions = roots_by_key.get("condition") or roots_by_key.get(
            "conditionExpression"
        )
        if not conditions:
            return
        bodies = []
        for key in (
            "TrueBody",
            "FalseBody",
            "body",
            "TrueExpression",
            "FalseExpression",
            "loopExpression",
        ):
            bodies.extend(roots_by_key.get(key, ()))
        for condition in conditions:
            for body in bodies:
                self._connect(graph, condition, body, "CONTROL")

    def _effect_edges(
        self,
        graph: MutableSemanticGraph,
        statements: list[dict[str, Any]],
        roots: list[int],
    ) -> None:
        effect_roots = [
            root
            for (statement, root) in zip(statements, roots)
            if self._is_effectful(statement)
        ]
        for left, right in zip(effect_roots, effect_roots[1:]):
            self._connect(graph, left, right, "EFFECT")

    def _is_effectful(self, value: Any) -> bool:
        if isinstance(value, list):
            return any((self._is_effectful(child) for child in value))
        if not isinstance(value, dict):
            return False
        node_type = value.get("type")
        if node_type in {
            "EmitStatement",
            "ReturnStatement",
            "ThrowStatement",
            "RevertStatement",
        }:
            return True
        if node_type == "BinaryOperation" and value.get("operator") in {
            "=",
            "+=",
            "-=",
            "*=",
            "/=",
            "%=",
            "|=",
            "&=",
            "^=",
        }:
            return True
        if node_type == "UnaryOperation" and value.get("operator") in {
            "++",
            "--",
            "delete",
        }:
            return True
        if node_type == "FunctionCall":
            return True
        return any((self._is_effectful(child) for child in value.values()))


ASSIGNMENT_OPERATORS = {
    "=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "|=",
    "&=",
    "^=",
    "<<=",
    ">>=",
}
UPDATE_OPERATORS = {"++", "--", "delete"}
UNIT_TYPES = {"FunctionDefinition", "ModifierDefinition", "StateVariableDeclaration"}
SymbolPath = tuple[int | str, ...]


@dataclass(frozen=True)
class Symbol:
    identifier: int
    name: str
    storage: str


class FlowEnvironment:
    def __init__(
        self,
        *,
        scopes: list[dict[str, int]] | None = None,
        definitions: dict[SymbolPath, set[int]] | None = None,
    ) -> None:
        self.scopes = [dict(scope) for scope in scopes or [{}]]
        self.definitions = {
            path: set(nodes) for (path, nodes) in (definitions or {}).items()
        }

    def clone(self) -> "FlowEnvironment":
        return FlowEnvironment(scopes=self.scopes, definitions=self.definitions)

    def push_scope(self) -> None:
        self.scopes.append({})

    def pop_scope(self) -> None:
        if len(self.scopes) == 1:
            raise ValueError("cannot remove root scope")
        self.scopes.pop()

    def bind(self, name: str, symbol: int) -> None:
        self.scopes[-1][name] = symbol

    def resolve(self, name: str) -> int | None:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def reaching(self, path: SymbolPath) -> set[int]:
        candidate = path
        while candidate:
            if candidate in self.definitions:
                return set(self.definitions[candidate])
            candidate = candidate[:-1]
        return set()

    def write(self, path: SymbolPath, node: int) -> None:
        self.definitions[path] = {node}
        if len(path) == 1:
            for child in [
                key for key in self.definitions if key[:1] == path and key != path
            ]:
                self.definitions.pop(child, None)

    def merge_definitions(self, *branches: "FlowEnvironment") -> None:
        keys = set(self.definitions)
        for branch in branches:
            keys.update(branch.definitions)
        self.definitions = {
            key: set().union(
                *(branch.definitions.get(key, set()) for branch in branches)
            )
            for key in keys
        }


class ScopeAwareDefUseAnalyzer:
    def __init__(self, ast_nodes: dict[int, int]) -> None:
        self.ast_nodes = ast_nodes
        self.symbols: dict[int, Symbol] = {}
        self.next_symbol = 0
        self.edges: list[tuple[int, int]] = []
        self.edge_set: set[tuple[int, int]] = set()
        self.stats: Counter[str] = Counter()
        self.unit_contract: dict[int, int] = {}
        self.contract_nodes: dict[int, dict[str, Any]] = {}
        self.contract_names: dict[str, int] = {}
        self.direct_states: dict[int, dict[str, int]] = {}
        self.state_definitions: dict[int, dict[SymbolPath, set[int]]] = {}

    def analyze(
        self, ast: dict[str, Any]
    ) -> tuple[list[tuple[int, int]], dict[str, int]]:
        self._index_contracts(ast)
        self._create_state_symbols()
        for contract_id, contract in self.contract_nodes.items():
            for unit in contract.get("subNodes") or ():
                if not isinstance(unit, dict):
                    continue
                node_type = unit.get("type")
                if node_type == "StateVariableDeclaration":
                    self._analyze_state_initializer(unit, contract_id)
                elif node_type in {"FunctionDefinition", "ModifierDefinition"}:
                    self._analyze_callable(unit, contract_id)
        self.stats["symbols"] = len(self.symbols)
        self.stats["data_edges_forward"] = len(self.edges)
        return (list(self.edges), dict(sorted(self.stats.items())))

    def _index_contracts(self, ast: Any) -> None:

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if value.get("type") == "ContractDefinition":
                    contract_id = id(value)
                    self.contract_nodes[contract_id] = value
                    name = str(
                        value.get("name") or f"contract_{len(self.contract_nodes)}"
                    )
                    self.contract_names[name] = contract_id
                    for unit in value.get("subNodes") or ():
                        if isinstance(unit, dict) and unit.get("type") in UNIT_TYPES:
                            self.unit_contract[id(unit)] = contract_id
                    return
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(ast)

    def _create_state_symbols(self) -> None:
        for contract_id, contract in self.contract_nodes.items():
            bindings: dict[str, int] = {}
            definitions: dict[SymbolPath, set[int]] = {}
            for unit in contract.get("subNodes") or ():
                if (
                    not isinstance(unit, dict)
                    or unit.get("type") != "StateVariableDeclaration"
                ):
                    continue
                for declaration in unit.get("variables") or ():
                    if not isinstance(declaration, dict):
                        continue
                    name = declaration.get("name")
                    node = self._node(declaration) or self._node(unit)
                    if not name or node is None:
                        continue
                    symbol = self._new_symbol(str(name), "state")
                    bindings[str(name)] = symbol
                    definitions[symbol,] = {node}
                    self.stats["state_definitions"] += 1
            self.direct_states[contract_id] = bindings
            self.state_definitions[contract_id] = definitions

    def _new_symbol(self, name: str, storage: str) -> int:
        identifier = self.next_symbol
        self.next_symbol += 1
        self.symbols[identifier] = Symbol(identifier, name, storage)
        return identifier

    def _environment(self, contract_id: int) -> FlowEnvironment:
        bindings = self._inherited_state_bindings(contract_id, set())
        definitions: dict[SymbolPath, set[int]] = {}
        for owner in self._inheritance_order(contract_id, set()):
            definitions.update(self.state_definitions.get(owner, {}))
        return FlowEnvironment(scopes=[bindings, {}], definitions=definitions)

    def _inheritance_order(self, contract_id: int, seen: set[int]) -> list[int]:
        if contract_id in seen:
            return []
        seen.add(contract_id)
        result: list[int] = []
        contract = self.contract_nodes[contract_id]
        for base in contract.get("baseContracts") or ():
            name = self._base_contract_name(base)
            owner = self.contract_names.get(name)
            if owner is not None:
                result.extend(self._inheritance_order(owner, seen))
        result.append(contract_id)
        return result

    def _inherited_state_bindings(
        self, contract_id: int, seen: set[int]
    ) -> dict[str, int]:
        bindings: dict[str, int] = {}
        for owner in self._inheritance_order(contract_id, seen):
            bindings.update(self.direct_states.get(owner, {}))
        return bindings

    @staticmethod
    def _base_contract_name(base: Any) -> str:
        if not isinstance(base, dict):
            return ""
        value = base.get("baseName", base)
        if isinstance(value, dict):
            for key in ("namePath", "name"):
                if value.get(key):
                    return str(value[key]).split(".")[-1]
        return ""

    def _analyze_state_initializer(
        self, unit: dict[str, Any], contract_id: int
    ) -> None:
        initial = unit.get("initialValue")
        if initial is None:
            return
        env = self._environment(contract_id)
        self._read_expression(initial, env)
        source = self._value_node(initial, env)
        for declaration in unit.get("variables") or ():
            if not isinstance(declaration, dict):
                continue
            name = declaration.get("name")
            symbol = self.direct_states.get(contract_id, {}).get(str(name))
            target = self._node(declaration)
            if symbol is None or target is None:
                continue
            if source is not None:
                self._add_edge(source, target, "value_flow")
            env.write((symbol,), target)

    def _analyze_callable(self, unit: dict[str, Any], contract_id: int) -> None:
        env = self._environment(contract_id)
        self.stats["callable_units"] += 1
        for key, storage in (
            ("parameters", "parameter"),
            ("returnParameters", "return"),
        ):
            parameter_list = unit.get(key)
            if not isinstance(parameter_list, dict):
                continue
            for declaration in parameter_list.get("parameters") or ():
                self._bind_declaration(declaration, env, storage=storage)
        for modifier in unit.get("modifiers") or ():
            if isinstance(modifier, dict):
                for argument in modifier.get("arguments") or ():
                    self._read_expression(argument, env)
        body = unit.get("body")
        if isinstance(body, dict):
            self._process(body, env)

    def _bind_declaration(
        self, declaration: Any, env: FlowEnvironment, *, storage: str = "local"
    ) -> int | None:
        if not isinstance(declaration, dict):
            return None
        name = declaration.get("name")
        node = self._node(declaration)
        if not name or node is None:
            return None
        symbol = self._new_symbol(str(name), storage)
        env.bind(str(name), symbol)
        env.write((symbol,), node)
        self.stats[f"{storage}_definitions"] += 1
        return node

    def _process(self, value: Any, env: FlowEnvironment) -> None:
        if value is None:
            return
        if isinstance(value, list):
            for child in value:
                self._process(child, env)
            return
        if not isinstance(value, dict):
            return
        node_type = value.get("type")
        if node_type == "Block":
            env.push_scope()
            for statement in value.get("statements") or ():
                self._process(statement, env)
            env.pop_scope()
            return
        if node_type == "VariableDeclarationStatement":
            self._variable_declaration_statement(value, env)
            return
        if node_type == "ExpressionStatement":
            self._read_expression(value.get("expression"), env)
            return
        if node_type == "IfStatement":
            self._if_statement(value, env)
            return
        if node_type == "ForStatement":
            self._for_statement(value, env)
            return
        if node_type == "WhileStatement":
            self._while_statement(value, env, body_first=False)
            return
        if node_type == "DoWhileStatement":
            self._while_statement(value, env, body_first=True)
            return
        if node_type in {
            "Identifier",
            "MemberAccess",
            "IndexAccess",
            "BinaryOperation",
            "UnaryOperation",
            "FunctionCall",
            "Conditional",
            "TupleExpression",
        }:
            self._read_expression(value, env)
            return
        if node_type in {"VariableDeclaration", "Parameter", "ParameterList"}:
            return
        for child in value.values():
            if isinstance(child, (dict, list)):
                self._process(child, env)

    def _variable_declaration_statement(
        self, statement: dict[str, Any], env: FlowEnvironment
    ) -> None:
        initial = statement.get("initialValue")
        if initial is not None:
            self._read_expression(initial, env)
        source = self._value_node(initial, env)
        for declaration in statement.get("variables") or ():
            target = self._bind_declaration(declaration, env)
            if source is not None and target is not None:
                self._add_edge(source, target, "value_flow")

    def _if_statement(self, statement: dict[str, Any], env: FlowEnvironment) -> None:
        self._read_expression(statement.get("condition"), env)
        incoming = env.clone()
        true_env = incoming.clone()
        self._process(statement.get("TrueBody"), true_env)
        false_env = incoming.clone()
        self._process(statement.get("FalseBody"), false_env)
        env.merge_definitions(true_env, false_env)

    def _for_statement(self, statement: dict[str, Any], env: FlowEnvironment) -> None:
        env.push_scope()
        self._process(statement.get("initExpression"), env)
        self._read_expression(statement.get("conditionExpression"), env)
        incoming = env.clone()
        body_env = incoming.clone()
        self._process(statement.get("body"), body_env)
        self._read_expression(statement.get("loopExpression"), body_env)
        env.merge_definitions(incoming, body_env)
        env.pop_scope()

    def _while_statement(
        self, statement: dict[str, Any], env: FlowEnvironment, *, body_first: bool
    ) -> None:
        incoming = env.clone()
        body_env = incoming.clone()
        if not body_first:
            self._read_expression(statement.get("condition"), body_env)
        self._process(statement.get("body"), body_env)
        if body_first:
            self._read_expression(statement.get("condition"), body_env)
        env.merge_definitions(incoming, body_env)

    def _read_expression(self, expression: Any, env: FlowEnvironment) -> None:
        if expression is None:
            return
        if isinstance(expression, list):
            for child in expression:
                self._read_expression(child, env)
            return
        if not isinstance(expression, dict):
            return
        node_type = expression.get("type")
        if node_type == "Identifier":
            self._read_access(expression, env)
            return
        if node_type in {"MemberAccess", "IndexAccess"}:
            path = self._access_path(expression, env)
            if path is not None:
                self._read_path(path, self._node(expression), env)
                self._read_index_components(expression, env)
            else:
                base = expression.get("expression") or expression.get("base")
                self._read_expression(base, env)
                self._read_expression(expression.get("index"), env)
            return
        if node_type == "BinaryOperation":
            operator = str(expression.get("operator", ""))
            if operator in ASSIGNMENT_OPERATORS:
                self._assignment(expression, env, operator)
            else:
                self._read_expression(expression.get("left"), env)
                self._read_expression(expression.get("right"), env)
            return
        if node_type == "UnaryOperation":
            operator = str(expression.get("operator", ""))
            child = expression.get("subExpression")
            if operator in UPDATE_OPERATORS:
                self._write_access(child, env, read_before=operator != "delete")
            else:
                self._read_expression(child, env)
            return
        if node_type == "FunctionCall":
            callee = expression.get("expression")
            if isinstance(callee, dict) and callee.get("type") == "MemberAccess":
                self._read_expression(callee.get("expression"), env)
            elif isinstance(callee, dict) and callee.get("type") != "Identifier":
                self._read_expression(callee, env)
            elif (
                isinstance(callee, dict) and self._access_path(callee, env) is not None
            ):
                self._read_expression(callee, env)
            for argument in expression.get("arguments") or ():
                self._read_expression(argument, env)
            return
        if node_type == "Conditional":
            self._read_expression(expression.get("condition"), env)
            incoming = env.clone()
            left = incoming.clone()
            self._read_expression(expression.get("TrueExpression"), left)
            right = incoming.clone()
            self._read_expression(expression.get("FalseExpression"), right)
            env.merge_definitions(left, right)
            return
        if node_type == "TupleExpression":
            self._read_expression(expression.get("components"), env)
            return
        for child in expression.values():
            if isinstance(child, (dict, list)):
                self._read_expression(child, env)

    def _assignment(
        self, expression: dict[str, Any], env: FlowEnvironment, operator: str
    ) -> None:
        right = expression.get("right")
        left = expression.get("left")
        self._read_expression(right, env)
        target_nodes = self._write_access(left, env, read_before=operator != "=")
        source = self._value_node(right, env)
        if source is not None:
            for target in target_nodes:
                self._add_edge(source, target, "value_flow")

    def _read_access(self, expression: dict[str, Any], env: FlowEnvironment) -> None:
        path = self._access_path(expression, env)
        if path is None:
            name = str(expression.get("name") or "")
            if name and name not in {"msg", "block", "tx", "this", "now"}:
                self.stats["unresolved_identifier_uses"] += 1
            return
        self._read_path(path, self._node(expression), env)

    def _read_path(
        self, path: SymbolPath, node: int | None, env: FlowEnvironment
    ) -> None:
        if node is None:
            return
        definitions = env.reaching(path)
        if not definitions:
            self.stats["uses_without_reaching_definition"] += 1
            return
        for definition in sorted(definitions):
            self._add_edge(definition, node, "def_use")
        self.stats["resolved_uses"] += 1

    def _write_access(
        self, expression: Any, env: FlowEnvironment, *, read_before: bool
    ) -> list[int]:
        if not isinstance(expression, dict):
            return []
        if expression.get("type") == "TupleExpression":
            result: list[int] = []
            for component in expression.get("components") or ():
                result.extend(
                    self._write_access(component, env, read_before=read_before)
                )
            return result
        path = self._access_path(expression, env)
        node = self._node(expression)
        if path is None or node is None:
            self._read_expression(expression, env)
            return []
        self._read_index_components(expression, env)
        if read_before:
            self._read_path(path, node, env)
        env.write(path, node)
        self.stats["write_definitions"] += 1
        return [node]

    def _access_path(self, expression: Any, env: FlowEnvironment) -> SymbolPath | None:
        if not isinstance(expression, dict):
            return None
        node_type = expression.get("type")
        if node_type == "Identifier":
            name = str(expression.get("name") or "")
            symbol = env.resolve(name)
            return (symbol,) if symbol is not None else None
        if node_type == "MemberAccess":
            base = expression.get("expression")
            member = str(expression.get("memberName") or "")
            if (
                isinstance(base, dict)
                and base.get("type") == "Identifier"
                and (base.get("name") == "this")
            ):
                symbol = env.resolve(member)
                return (symbol,) if symbol is not None else None
            base_path = self._access_path(base, env)
            return (*base_path, f"member:{member}") if base_path is not None else None
        if node_type == "IndexAccess":
            base_path = self._access_path(expression.get("base"), env)
            return (*base_path, "index:*") if base_path is not None else None
        return None

    def _read_index_components(self, expression: Any, env: FlowEnvironment) -> None:
        if not isinstance(expression, dict):
            return
        node_type = expression.get("type")
        if node_type == "IndexAccess":
            self._read_expression(expression.get("index"), env)
            self._read_index_components(expression.get("base"), env)
        elif node_type == "MemberAccess":
            self._read_index_components(expression.get("expression"), env)

    def _add_edge(self, source: int, target: int, kind: str) -> None:
        if source == target or (source, target) in self.edge_set:
            return
        self.edge_set.add((source, target))
        self.edges.append((source, target))
        self.stats[f"{kind}_edges"] += 1

    def _node(self, value: Any) -> int | None:
        return self.ast_nodes.get(id(value)) if isinstance(value, dict) else None

    def _value_node(self, value: Any, env: FlowEnvironment) -> int | None:
        if not isinstance(value, dict):
            return None
        if not self._has_data_dependency(value, env):
            return None
        return self._node(value)

    def _has_data_dependency(self, value: Any, env: FlowEnvironment) -> bool:
        if value is None:
            return False
        if isinstance(value, list):
            return any((self._has_data_dependency(child, env) for child in value))
        if not isinstance(value, dict):
            return False
        node_type = str(value.get("type", ""))
        if node_type.endswith("Literal"):
            return False
        if node_type == "Identifier":
            name = str(value.get("name") or "")
            return env.resolve(name) is not None or name in {
                "msg",
                "block",
                "tx",
                "now",
            }
        if node_type in {"MemberAccess", "IndexAccess"}:
            return self._access_path(value, env) is not None or any(
                (
                    self._has_data_dependency(value.get(key), env)
                    for key in ("expression", "base", "index")
                )
            )
        if node_type in {"FunctionCall", "NewExpression"}:
            return True
        return any(
            (
                self._has_data_dependency(child, env)
                for child in value.values()
                if isinstance(child, (dict, list))
            )
        )


class ScopeAwareDefUseGraphBuilder(SoliditySemanticGraphBuilder):
    def __init__(self):
        super().__init__()
        self._ast_graph_nodes = {}

    def _collect_functions(self, value: Any) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []

        def visit(current: Any) -> None:
            if isinstance(current, dict):
                if current.get("type") == "ContractDefinition":
                    units.extend(
                        (
                            child
                            for child in current.get("subNodes") or ()
                            if isinstance(child, dict)
                            and child.get("type") in UNIT_TYPES
                        )
                    )
                    return
                for child in current.values():
                    visit(child)
            elif isinstance(current, list):
                for child in current:
                    visit(child)

        visit(value)
        return units

    def _visit_dict(
        self,
        value: dict[str, Any],
        parent: int,
        field_name: str,
        graph: MutableSemanticGraph,
        identity_nodes: list[tuple[str, int]],
    ) -> int:
        node = super()._visit_dict(value, parent, field_name, graph, identity_nodes)
        self._ast_graph_nodes[id(value)] = node
        return node


COMPARISON_OPERATORS = {"<", "<=", ">", ">=", "==", "!=", "&&", "||"}
CONTROL_TYPES = {
    "IfStatement",
    "ForStatement",
    "WhileStatement",
    "DoWhileStatement",
    "Conditional",
}
EFFECT_TYPES = {
    "ReturnStatement",
    "EmitStatement",
    "ThrowStatement",
    "BreakStatement",
    "ContinueStatement",
    "NewExpression",
}
ENVIRONMENT_ROOTS = {"block", "msg", "tx"}


class RoleAwareDefUseAnalyzer(ScopeAwareDefUseAnalyzer):
    def __init__(self, ast_nodes: dict[int, int]) -> None:
        super().__init__(ast_nodes)
        self.roles: dict[int, set[str]] = defaultdict(set)

    def _create_state_symbols(self) -> None:
        super()._create_state_symbols()
        for definitions in self.state_definitions.values():
            for nodes in definitions.values():
                for node in nodes:
                    self.roles[node].add("state_def")

    def _bind_declaration(
        self, declaration: Any, env: Any, *, storage: str = "local"
    ) -> int | None:
        node = super()._bind_declaration(declaration, env, storage=storage)
        if node is not None:
            self.roles[node].add(f"{storage}_def")
        return node

    def _read_path(self, path: SymbolPath, node: int | None, env: Any) -> None:
        if node is not None and path:
            symbol = self.symbols.get(int(path[0]))
            if symbol is not None:
                self.roles[node].add(f"{symbol.storage}_read")
        super()._read_path(path, node, env)

    def _write_access(
        self, expression: Any, env: Any, *, read_before: bool
    ) -> list[int]:
        path = self._access_path(expression, env)
        node = self._node(expression)
        result = super()._write_access(expression, env, read_before=read_before)
        if path and node is not None and (node in result):
            symbol = self.symbols.get(int(path[0]))
            if symbol is not None:
                self.roles[node].add(f"{symbol.storage}_write")
        return result


def _children(graph: SemanticGraph) -> list[list[int]]:
    result = [[] for _ in range(graph.num_nodes)]
    for node, parent in enumerate(graph.parent):
        if parent >= 0:
            result[parent].append(node)
    return result


def _subtree_scalars(
    graph: SemanticGraph, children: list[list[int]], root: int
) -> list[str]:
    values: list[str] = []
    pending = [root]
    while pending:
        node = pending.pop()
        if graph.node_kinds[node] in {"identifier", "operator", "attribute"}:
            values.append(str(graph.raw_tokens[node]).lower())
        pending.extend(reversed(children[node]))
    return values


def _operator(graph: SemanticGraph, children: list[list[int]], node: int) -> str:
    for child in children[node]:
        if graph.node_kinds[child] == "operator":
            return str(graph.raw_tokens[child])
    return "unknown"


def _environment_source(
    graph: SemanticGraph, children: list[list[int]], node: int
) -> str | None:
    raw_type = graph.raw_tokens[node]
    values = _subtree_scalars(graph, children, node)
    if raw_type == "Identifier" and "now" in values:
        return "now"
    if raw_type not in {"MemberAccess", "IndexAccess"}:
        return None
    roots = [root for root in ENVIRONMENT_ROOTS if root in values]
    if not roots:
        return None
    root = roots[0]
    members = [value for value in values if value != root and value not in {"this"}]
    return ".".join((root, members[-1])) if members else root


def _call_kind(graph: SemanticGraph, children: list[list[int]], node: int) -> str:
    values = _subtree_scalars(graph, children, node)
    for candidate in (
        "delegatecall",
        "callcode",
        "selfdestruct",
        "transfer",
        "require",
        "assert",
        "revert",
        "send",
        "call",
        "keccak256",
        "sha256",
    ):
        if candidate in values:
            return candidate
    return "user"


def _entry_token(graph: SemanticGraph, entry: int) -> str:
    for node, parent in enumerate(graph.parent):
        if parent == entry and graph.node_kinds[node] == "ast_type":
            raw = str(graph.raw_tokens[node]).replace("Definition", "").lower()
            return f"semantic:unit:{raw}"
    return "semantic:unit:unknown"


def retained_node_tokens(
    graph: SemanticGraph, roles: dict[int, set[str]]
) -> dict[int, str]:
    children = _children(graph)
    result: dict[int, str] = {}
    for node in range(graph.num_nodes):
        if graph.parent[node] < 0:
            result[node] = _entry_token(graph, node)
            continue
        if graph.node_kinds[node] != "ast_type":
            continue
        local_roles = roles.get(node, set())
        state_roles = sorted(
            (role for role in local_roles if role.startswith("state_"))
        )
        if state_roles:
            suffix = "+".join((role.removeprefix("state_") for role in state_roles))
            result[node] = f"semantic:state:{suffix}"
            continue
        if "parameter_def" in local_roles:
            result[node] = "semantic:input:parameter"
            continue
        source = _environment_source(graph, children, node)
        if source is not None:
            result[node] = f"semantic:input:{source}"
            continue
        raw_type = str(graph.raw_tokens[node])
        if raw_type == "FunctionCall":
            result[node] = f"semantic:call:{_call_kind(graph, children, node)}"
        elif raw_type == "BinaryOperation":
            operator = _operator(graph, children, node)
            if operator in COMPARISON_OPERATORS:
                result[node] = f"semantic:compare:{operator}"
        elif raw_type in CONTROL_TYPES:
            result[node] = f"semantic:control:{raw_type.lower()}"
        elif raw_type in EFFECT_TYPES:
            result[node] = f"semantic:effect:{raw_type.lower()}"
    return result


EDGE_TYPES = (
    "SEMANTIC_NEXT",
    "SEMANTIC_NEXT_REV",
    "AST_CONTROL",
    "AST_CONTROL_REV",
    "CFG_SUMMARY_NEXT",
    "CFG_SUMMARY_NEXT_REV",
    "CFG_SUMMARY_TRUE",
    "CFG_SUMMARY_TRUE_REV",
    "CFG_SUMMARY_FALSE",
    "CFG_SUMMARY_FALSE_REV",
    "CFG_SUMMARY_BACK",
    "CFG_SUMMARY_BACK_REV",
    "VALUE_FLOW",
    "VALUE_FLOW_REV",
)
EDGE_TYPE_TO_ID = {name: index for (index, name) in enumerate(EDGE_TYPES)}
REVERSE_RELATION = {
    "SEMANTIC_NEXT": "SEMANTIC_NEXT_REV",
    "AST_CONTROL": "AST_CONTROL_REV",
    "CFG_SUMMARY_NEXT": "CFG_SUMMARY_NEXT_REV",
    "CFG_SUMMARY_TRUE": "CFG_SUMMARY_TRUE_REV",
    "CFG_SUMMARY_FALSE": "CFG_SUMMARY_FALSE_REV",
    "CFG_SUMMARY_BACK": "CFG_SUMMARY_BACK_REV",
    "VALUE_FLOW": "VALUE_FLOW_REV",
}
CFG_NEXT = "CFG_NEXT"
CFG_TRUE = "CFG_TRUE"
CFG_FALSE = "CFG_FALSE"
CFG_BACK = "CFG_BACK"


@dataclass(frozen=True)
class CFGNode:
    key: str
    unit: int
    kind: str
    ast_identity: int | None = None


@dataclass(frozen=True)
class CFGEdge:
    source: str
    target: str
    kind: str


@dataclass
class StructuredCFG:
    nodes: dict[str, CFGNode] = field(default_factory=dict)
    edges: list[CFGEdge] = field(default_factory=list)
    entry_by_unit: dict[int, str] = field(default_factory=dict)
    exit_by_unit: dict[int, str] = field(default_factory=dict)
    ast_to_cfg: dict[int, str] = field(default_factory=dict)
    predicate_keys: set[str] = field(default_factory=set)
    defuse_available: bool = False

    def outgoing(self) -> dict[str, list[CFGEdge]]:
        result: dict[str, list[CFGEdge]] = defaultdict(list)
        for edge in self.edges:
            result[edge.source].append(edge)
        return result


class SlitherCFGBuilder:
    def __init__(
        self, source_path: str | Path, *, compiler_version: str, source: str
    ) -> None:
        self.source_path = Path(source_path).resolve()
        self.compiler_version = compiler_version
        self.source = source
        self.line_offsets = self._line_offsets(source)

    def build(self, units: Iterable[dict[str, Any]]) -> StructuredCFG:
        previous = os.environ.get("SOLC_VERSION")
        os.environ["SOLC_VERSION"] = self.compiler_version
        try:
            try:
                slither = Slither(str(self.source_path), skip_analyze=False)
                defuse_available = True
            except Exception:
                slither = Slither(str(self.source_path), skip_analyze=True)
                defuse_available = False
        finally:
            if previous is None:
                os.environ.pop("SOLC_VERSION", None)
            else:
                os.environ["SOLC_VERSION"] = previous
        cfg = StructuredCFG(defuse_available=defuse_available)
        source_functions = []
        for contract in slither.contracts:
            source_functions.extend(
                (
                    function
                    for function in contract.functions_declared
                    if self._belongs_to_source(function)
                )
            )
            source_functions.extend(
                (
                    modifier
                    for modifier in contract.modifiers_declared
                    if self._belongs_to_source(modifier)
                )
            )
        for unit, ast_unit in enumerate(units):
            unit_type = str(ast_unit.get("type", ""))
            entry = f"u{unit}:entry"
            exit_node = f"u{unit}:exit"
            cfg.nodes[entry] = CFGNode(entry, unit, "ENTRY", id(ast_unit))
            cfg.nodes[exit_node] = CFGNode(exit_node, unit, "EXIT")
            cfg.entry_by_unit[unit] = entry
            cfg.exit_by_unit[unit] = exit_node
            cfg.ast_to_cfg[id(ast_unit)] = entry
            if unit_type == "StateVariableDeclaration":
                declaration = f"u{unit}:state_declaration"
                cfg.nodes[declaration] = CFGNode(
                    declaration, unit, "STATE_DECLARATION", id(ast_unit)
                )
                self._map_ast_tree(ast_unit, declaration, cfg)
                cfg.edges.extend(
                    (
                        CFGEdge(entry, declaration, CFG_NEXT),
                        CFGEdge(declaration, exit_node, CFG_NEXT),
                    )
                )
                continue
            function = self._match_function(ast_unit, source_functions)
            if function is None:
                raise ValueError(
                    f"cannot map AST unit {ast_unit.get('name')!r} to Slither CFG"
                )
            slither_nodes = list(function.nodes)
            key_by_node: dict[Any, str] = {}
            for local, node in enumerate(slither_nodes):
                key = (
                    entry
                    if node.type == NodeType.ENTRYPOINT
                    else f"u{unit}:cfg:{local}"
                )
                key_by_node[node] = key
                if key == entry:
                    continue
                kind = str(node.type.value)
                cfg.nodes[key] = CFGNode(key, unit, kind)
                if node.type in {NodeType.IF, NodeType.IFLOOP}:
                    cfg.predicate_keys.add(key)
            if not slither_nodes:
                cfg.edges.append(CFGEdge(entry, exit_node, CFG_NEXT))
                self._map_unit_ast(unit, ast_unit, function, key_by_node, cfg)
                continue
            if not any((node.type == NodeType.ENTRYPOINT for node in slither_nodes)):
                cfg.edges.append(
                    CFGEdge(entry, key_by_node[slither_nodes[0]], CFG_NEXT)
                )
            for node in slither_nodes:
                source_key = key_by_node[node]
                if not node.sons:
                    cfg.edges.append(CFGEdge(source_key, exit_node, CFG_NEXT))
                    continue
                for target in node.sons:
                    cfg.edges.append(
                        CFGEdge(
                            source_key,
                            key_by_node[target],
                            self._edge_kind(node, target),
                        )
                    )
            self._map_unit_ast(unit, ast_unit, function, key_by_node, cfg)
        cfg.edges = list(dict.fromkeys(cfg.edges))
        return cfg

    def _belongs_to_source(self, declaration: Any) -> bool:
        filename = str(declaration.source_mapping.filename.absolute or "")
        return not filename or Path(filename).resolve() == self.source_path

    def _match_function(self, ast_unit: dict[str, Any], functions: list[Any]) -> Any:
        interval = self._ast_interval(ast_unit)
        if interval is None:
            return None
        (start, end) = interval
        name = str(ast_unit.get("name") or "")
        candidates = []
        for function in functions:
            mapping = function.source_mapping
            function_start = int(mapping.start)
            function_end = function_start + int(mapping.length)
            overlaps = (
                function_start <= start <= function_end
                or start <= function_start <= end
            )
            if not overlaps:
                continue
            name_penalty = 0 if function.name == name else 1
            candidates.append(
                (
                    abs(function_start - start),
                    name_penalty,
                    int(mapping.length),
                    function,
                )
            )
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[:3])[-1]

    def _map_unit_ast(
        self,
        unit: int,
        ast_unit: dict[str, Any],
        function: Any,
        key_by_node: dict[Any, str],
        cfg: StructuredCFG,
    ) -> None:
        intervals = []
        for node in function.nodes:
            mapping = node.source_mapping
            start = int(mapping.start)
            length = max(int(mapping.length), 1)
            intervals.append(
                (
                    start,
                    start + length,
                    length,
                    1 if node.type == NodeType.ENTRYPOINT else 0,
                    key_by_node[node],
                )
            )

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                interval = self._ast_interval(value)
                if interval is not None:
                    (start, _) = interval
                    candidates = [
                        item for item in intervals if item[0] <= start < item[1]
                    ]
                    if candidates:
                        cfg.ast_to_cfg[id(value)] = min(
                            candidates, key=lambda item: (item[3], item[2])
                        )[4]
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(ast_unit)
        cfg.ast_to_cfg[id(ast_unit)] = cfg.entry_by_unit[unit]

    def _map_ast_tree(self, value: Any, target: str, cfg: StructuredCFG) -> None:
        if isinstance(value, dict):
            cfg.ast_to_cfg[id(value)] = target
            for child in value.values():
                self._map_ast_tree(child, target, cfg)
        elif isinstance(value, list):
            for child in value:
                self._map_ast_tree(child, target, cfg)

    def _ast_interval(self, value: dict[str, Any]) -> tuple[int, int] | None:
        location = value.get("loc")
        if not isinstance(location, dict):
            return None
        start = location.get("start")
        end = location.get("end")
        if not isinstance(start, dict) or not isinstance(end, dict):
            return None
        start_offset = self._offset(int(start["line"]), int(start["column"]))
        end_offset = self._offset(int(end["line"]), int(end["column"])) + 1
        return (start_offset, max(start_offset + 1, end_offset))

    def _offset(self, line: int, column: int) -> int:
        line_index = max(line - 1, 0)
        if line_index >= len(self.line_offsets):
            return len(self.source.encode("utf-8"))
        prefix = self.source[: self.line_offsets[line_index] + column]
        return len(prefix.encode("utf-8"))

    @staticmethod
    def _line_offsets(source: str) -> list[int]:
        offsets = [0]
        for index, char in enumerate(source):
            if char == "\n":
                offsets.append(index + 1)
        return offsets

    @staticmethod
    def _edge_kind(source: Any, target: Any) -> str:
        if source.type in {NodeType.IF, NodeType.IFLOOP}:
            if source.son_true is target:
                return CFG_TRUE
            if source.son_false is target:
                return CFG_FALSE
        source_start = int(source.source_mapping.start)
        target_start = int(target.source_mapping.start)
        if source.type in {NodeType.CONTINUE, NodeType.ENDLOOP} or (
            target.type in {NodeType.STARTLOOP, NodeType.IFLOOP}
            and target_start <= source_start
        ):
            return CFG_BACK
        return CFG_NEXT


IDENTIFIER_PATTERN = re.compile("[A-Za-z_][A-Za-z0-9_]*")
CFG_SUMMARY_RELATION = {
    CFG_NEXT: "CFG_SUMMARY_NEXT",
    CFG_TRUE: "CFG_SUMMARY_TRUE",
    CFG_FALSE: "CFG_SUMMARY_FALSE",
    CFG_BACK: "CFG_SUMMARY_BACK",
}


def _add_bidirectional(
    edges: set[tuple[int, int, int]], source: int, target: int, relation: str
) -> None:
    if source == target:
        return
    edges.add((source, target, EDGE_TYPE_TO_ID[relation]))
    edges.add((target, source, EDGE_TYPE_TO_ID[REVERSE_RELATION[relation]]))


def _merge_path_kind(current: str, following: str) -> str:
    if current == CFG_BACK or following == CFG_BACK:
        return CFG_BACK
    if current in {CFG_TRUE, CFG_FALSE}:
        return current
    if following in {CFG_TRUE, CFG_FALSE}:
        return following
    return CFG_NEXT


class TypedRoleAwareDefUseAnalyzer(RoleAwareDefUseAnalyzer):
    def __init__(self, ast_nodes: dict[int, int]) -> None:
        super().__init__(ast_nodes)
        self.typed_edges: list[tuple[int, int, str]] = []

    def _add_edge(self, source: int, target: int, kind: str) -> None:
        already_present = source == target or (source, target) in self.edge_set
        super()._add_edge(source, target, kind)
        if not already_present:
            self.typed_edges.append((source, target, kind))


def _nearest_ast_parent(graph: SemanticGraph, node: int) -> int | None:
    parent = graph.parent[node]
    while parent >= 0:
        if graph.node_kinds[parent] == "ast_type":
            return parent
        parent = graph.parent[parent]
    return None


def _nearest_control_ancestor(
    graph: SemanticGraph, node: int, retained_nodes: set[int]
) -> int | None:
    parent = graph.parent[node]
    while parent >= 0:
        if parent in retained_nodes and graph.raw_tokens[parent] in CONTROL_TYPES:
            return parent
        parent = graph.parent[parent]
    return None


def _project_nodes(
    graph: SemanticGraph, tokens_by_origin: dict[int, str], *, expression_roles: bool
) -> SemanticGraph:
    origins = sorted(tokens_by_origin)
    origin_to_new = {origin: index for (index, origin) in enumerate(origins)}
    retained_nodes = set(origins)
    edges: set[tuple[int, int, int]] = set()
    for target in origins:
        source = _nearest_control_ancestor(graph, target, retained_nodes)
        if source is not None and source != target:
            _add_bidirectional(
                edges, origin_to_new[source], origin_to_new[target], "AST_CONTROL"
            )
    by_unit: dict[int, list[int]] = defaultdict(list)
    for origin in origins:
        if graph.parent[origin] >= 0:
            by_unit[graph.node_unit[origin]].append(origin)
    for local_origins in by_unit.values():
        for source, target in zip(local_origins, local_origins[1:]):
            _add_bidirectional(
                edges, origin_to_new[source], origin_to_new[target], "SEMANTIC_NEXT"
            )
    entry_by_unit = {
        graph.node_unit[origin]: origin_to_new[origin]
        for origin in origins
        if graph.parent[origin] < 0
    }
    parent = [
        -1
        if graph.parent[origin] < 0
        else entry_by_unit.get(graph.node_unit[origin], -1)
        for origin in origins
    ]
    ordered = sorted(edges, key=lambda edge: (edge[0], edge[2], edge[1]))
    return SemanticGraph(
        address=graph.address,
        domain=graph.domain,
        source_path=graph.source_path,
        label=graph.label,
        tokens=[tokens_by_origin[origin] for origin in origins],
        raw_tokens=[graph.raw_tokens[origin] for origin in origins],
        node_kinds=["semantic_anchor"] * len(origins),
        parent=parent,
        node_unit=[graph.node_unit[origin] for origin in origins],
        unit_names=list(graph.unit_names),
        edge_src=[source for (source, _, _) in ordered],
        edge_dst=[target for (_, target, _) in ordered],
        edge_type=[relation for (_, _, relation) in ordered],
        split=graph.split,
        metadata={
            "extractor": "semantic_expression_valueflow_graph_v1",
            "projection": "compact_role_aware_semantic_projection",
            "origin_nodes": origins,
            "expression_roles": expression_roles,
            "domain_used_during_construction": False,
            "transformation_rules_used": False,
        },
    )


def _iter_ast_values(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_ast_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_ast_values(child)


def _source_fragment(source: str, location: dict[str, Any]) -> str:
    lines = source.splitlines(keepends=True)
    start = location.get("start") or {}
    end = location.get("end") or {}
    start_line = int(start.get("line", 1)) - 1
    end_line = int(end.get("line", 1)) - 1
    if not (0 <= start_line < len(lines) and 0 <= end_line < len(lines)):
        return ""
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    first = offsets[start_line] + int(start.get("column", 0))
    last = offsets[end_line] + int(end.get("column", 0)) + 1
    return source[first:last]


def _split_tuple_components(value: str) -> list[str]:
    result: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(value):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth = max(depth - 1, 0)
        elif character == "," and depth == 0:
            result.append(value[start:index])
            start = index + 1
    result.append(value[start:])
    return result


def _recover_tuple_declarations(ast: dict[str, Any], source: str) -> int:
    recovered = 0
    for value in _iter_ast_values(ast):
        if value.get("type") != "VariableDeclarationStatement":
            continue
        if value.get("variables") is not None:
            continue
        fragment = _source_fragment(source, value.get("loc") or {}).strip()
        if not fragment.startswith("("):
            continue
        depth = 0
        closing = None
        for index, character in enumerate(fragment):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    closing = index
                    break
        if closing is None:
            continue
        declarations: list[dict[str, str] | None] = []
        for component in _split_tuple_components(fragment[1:closing]):
            names = IDENTIFIER_PATTERN.findall(component)
            if len(names) < 2:
                declarations.append(None)
                continue
            declarations.append({"type": "VariableDeclaration", "name": names[-1]})
            recovered += 1
        value["variables"] = declarations
    return recovered


def _value_consumption_edges(
    ast: dict[str, Any], ast_nodes: dict[int, int]
) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()

    def node(value: Any) -> int | None:
        return ast_nodes.get(id(value)) if isinstance(value, dict) else None

    def add(child: Any, parent: dict[str, Any]) -> None:
        source = node(child)
        target = node(parent)
        if source is not None and target is not None and (source != target):
            result.add((source, target))

    for value in _iter_ast_values(ast):
        node_type = value.get("type")
        if node_type == "BinaryOperation":
            operator = str(value.get("operator") or "")
            if operator not in {
                "=",
                "+=",
                "-=",
                "*=",
                "/=",
                "%=",
                "|=",
                "&=",
                "^=",
                "<<=",
                ">>=",
            }:
                add(value.get("left"), value)
                add(value.get("right"), value)
        elif node_type == "UnaryOperation":
            add(value.get("subExpression"), value)
        elif node_type == "MemberAccess":
            add(value.get("expression"), value)
        elif node_type == "IndexAccess":
            add(value.get("base"), value)
            add(value.get("index"), value)
        elif node_type == "FunctionCall":
            callee = value.get("expression")
            if isinstance(callee, dict) and callee.get("type") == "MemberAccess":
                add(callee, value)
            for argument in value.get("arguments") or ():
                add(argument, value)
        elif node_type == "Conditional":
            add(value.get("condition"), value)
            add(value.get("TrueExpression"), value)
            add(value.get("FalseExpression"), value)
        elif node_type == "TupleExpression":
            for component in value.get("components") or ():
                add(component, value)
        elif node_type in {"IfStatement", "WhileStatement", "DoWhileStatement"}:
            add(value.get("condition"), value)
        elif node_type == "ForStatement":
            add(value.get("conditionExpression"), value)
        elif node_type == "ReturnStatement":
            add(value.get("expression"), value)
        elif node_type == "EmitStatement":
            add(value.get("eventCall"), value)
    return result


def _contract_value_flow(
    origins: list[int],
    raw_node_count: int,
    typed_edges: Iterable[tuple[int, int, str]],
    consumption_edges: Iterable[tuple[int, int]],
) -> set[tuple[int, int]]:
    adjacency = [set() for _ in range(raw_node_count)]
    for source, target, _kind in typed_edges:
        adjacency[source].add(target)
    for source, target in consumption_edges:
        adjacency[source].add(target)
    retained_nodes = set(origins)
    result: set[tuple[int, int]] = set()
    for source in origins:
        pending = deque(adjacency[source])
        visited = {source}
        while pending:
            target = pending.popleft()
            if target in visited:
                continue
            visited.add(target)
            if target in retained_nodes:
                if target != source:
                    result.add((source, target))
                continue
            pending.extend(adjacency[target])
    return result


class SCVGraphBuilder(ScopeAwareDefUseGraphBuilder):
    def __init__(self) -> None:
        super().__init__()
        self._graph_node_to_ast_identity: dict[int, int] = {}
        self._graph_node_to_ast_value: dict[int, dict[str, Any]] = {}

    def _visit_dict(
        self,
        value: dict[str, Any],
        parent: int,
        field_name: str,
        graph: Any,
        identity_nodes: list[tuple[str, int]],
    ) -> int:
        node = super()._visit_dict(value, parent, field_name, graph, identity_nodes)
        self._graph_node_to_ast_identity[node] = id(value)
        self._graph_node_to_ast_value[node] = value
        return node

    def _origin_cfg_key(
        self, raw_graph: SemanticGraph, cfg: StructuredCFG, origin: int
    ) -> str:
        if raw_graph.parent[origin] < 0:
            return cfg.entry_by_unit[raw_graph.node_unit[origin]]
        cursor = origin
        while cursor >= 0:
            ast_value = self._graph_node_to_ast_value.get(cursor)
            if ast_value is not None:
                condition = ast_value.get(
                    "conditionExpression"
                    if ast_value.get("type") == "ForStatement"
                    else "condition"
                )
                if isinstance(condition, dict):
                    condition_key = cfg.ast_to_cfg.get(id(condition))
                    if condition_key in cfg.predicate_keys:
                        return condition_key
            ast_identity = self._graph_node_to_ast_identity.get(cursor)
            if ast_identity is not None and ast_identity in cfg.ast_to_cfg:
                return cfg.ast_to_cfg[ast_identity]
            cursor = raw_graph.parent[cursor]
        return cfg.entry_by_unit[raw_graph.node_unit[origin]]

    @staticmethod
    def _add_cfg_summary(
        cfg: StructuredCFG,
        by_cfg: dict[str, list[int]],
        edges: set[tuple[int, int, int]],
    ) -> None:
        for local_nodes in by_cfg.values():
            for source, target in zip(local_nodes, local_nodes[1:]):
                _add_bidirectional(edges, source, target, "CFG_SUMMARY_NEXT")
        outgoing = cfg.outgoing()
        for cfg_source, local_nodes in by_cfg.items():
            source = local_nodes[-1]
            for first_edge in outgoing.get(cfg_source, ()):
                pending = deque([(first_edge.target, first_edge.kind)])
                visited: set[tuple[str, str]] = set()
                while pending:
                    (cfg_target, path_kind) = pending.popleft()
                    state = (cfg_target, path_kind)
                    if state in visited:
                        continue
                    visited.add(state)
                    target_nodes = by_cfg.get(cfg_target)
                    if target_nodes:
                        _add_bidirectional(
                            edges,
                            source,
                            target_nodes[0],
                            CFG_SUMMARY_RELATION[path_kind],
                        )
                        continue
                    for following in outgoing.get(cfg_target, ()):
                        pending.append(
                            (
                                following.target,
                                _merge_path_kind(path_kind, following.kind),
                            )
                        )

    def build(self, source_path, compiler_version="0.5.17"):
        path = Path(source_path)
        source = path.read_text(encoding="utf-8")
        ast = parser.parse(source, loc=True)
        _recover_tuple_declarations(ast, source)
        self._ast_graph_nodes = {}
        self._graph_node_to_ast_identity = {}
        self._graph_node_to_ast_value = {}
        raw = SoliditySemanticGraphBuilder.from_ast(
            self,
            ast,
            address=path.stem,
            domain="clean",
            source_path=str(path),
            label=(0, 0, 0),
            split="",
        )
        analyzer = TypedRoleAwareDefUseAnalyzer(self._ast_graph_nodes)
        analyzer.analyze(ast)
        consumption = _value_consumption_edges(ast, self._ast_graph_nodes)
        cfg = SlitherCFGBuilder(
            path, compiler_version=compiler_version, source=source
        ).build(self._collect_functions(ast) or [ast])
        graph = _project_nodes(
            raw, retained_node_tokens(raw, analyzer.roles), expression_roles=False
        )
        origins = graph.metadata["origin_nodes"]
        edges = set(zip(graph.edge_src, graph.edge_dst, graph.edge_type))
        by_cfg = defaultdict(list)
        for node, origin in enumerate(origins):
            by_cfg[self._origin_cfg_key(raw, cfg, origin)].append(node)
        self._add_cfg_summary(cfg, by_cfg, edges)
        origin_to_new = {origin: index for (index, origin) in enumerate(origins)}
        for source, target in _contract_value_flow(
            origins, raw.num_nodes, analyzer.typed_edges, consumption
        ):
            _add_bidirectional(
                edges, origin_to_new[source], origin_to_new[target], "VALUE_FLOW"
            )
        ordered = sorted(edges, key=lambda edge: (edge[0], edge[2], edge[1]))
        graph.edge_src = [source for (source, _, _) in ordered]
        graph.edge_dst = [target for (_, target, _) in ordered]
        graph.edge_type = [relation for (_, _, relation) in ordered]
        graph.metadata = {
            "variant": "valueflow_only",
            "source_cfg": "slither_compiler_cfg",
            "origin_nodes": origins,
        }
        return graph


@dataclass
class ProgramGraphBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    node_unit: torch.Tensor
    unit_graph: torch.Tensor
    node_graph: torch.Tensor
    labels: torch.Tensor
    graph_addresses: list[str]
    graph_domains: list[str]
    num_graphs: int
    num_units: int

    def to(self, device: torch.device | str) -> "ProgramGraphBatch":
        return ProgramGraphBatch(
            **{
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for (name, value) in self.__dict__.items()
            }
        )


def collate_graphs(
    graphs: Sequence[SemanticGraph], *, vocab_size: int = 32768
) -> ProgramGraphBatch:
    if not graphs:
        raise ValueError("cannot collate an empty graph list")
    x_parts: list[torch.Tensor] = []
    edge_parts: list[torch.Tensor] = []
    edge_type_parts: list[torch.Tensor] = []
    node_unit_parts: list[torch.Tensor] = []
    node_graph_parts: list[torch.Tensor] = []
    unit_graph_values: list[int] = []
    node_offset = 0
    unit_offset = 0
    for graph_index, graph in enumerate(graphs):
        x_parts.append(graph.token_ids(vocab_size))
        edge_parts.append(graph.edge_index() + node_offset)
        edge_type_parts.append(graph.edge_types())
        node_unit_parts.append(graph.node_units() + unit_offset)
        node_graph_parts.append(
            torch.full((graph.num_nodes,), graph_index, dtype=torch.long)
        )
        unit_graph_values.extend([graph_index] * graph.num_units)
        node_offset += graph.num_nodes
        unit_offset += graph.num_units
    return ProgramGraphBatch(
        x=torch.cat(x_parts),
        edge_index=torch.cat(edge_parts, dim=1),
        edge_type=torch.cat(edge_type_parts),
        node_unit=torch.cat(node_unit_parts),
        unit_graph=torch.tensor(unit_graph_values, dtype=torch.long),
        node_graph=torch.cat(node_graph_parts),
        labels=torch.tensor([graph.label for graph in graphs], dtype=torch.long),
        graph_addresses=[graph.address for graph in graphs],
        graph_domains=[graph.domain for graph in graphs],
        num_graphs=len(graphs),
        num_units=unit_offset,
    )


def segment_softmax(
    scores: torch.Tensor, segments: torch.Tensor, num_segments: int
) -> torch.Tensor:
    work = scores.float()
    maxima = work.new_full((num_segments,), -torch.inf)
    maxima.scatter_reduce_(0, segments, work, reduce="amax", include_self=True)
    exponentials = torch.exp(work - maxima[segments])
    denominators = work.new_zeros(num_segments)
    denominators.index_add_(0, segments, exponentials)
    return (exponentials / denominators[segments].clamp_min(1e-12)).to(scores.dtype)


class HierarchicalAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.node_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.unit_norm = nn.LayerNorm(hidden_dim)
        self.unit_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor, batch: ProgramGraphBatch) -> torch.Tensor:
        node_scores = self.node_score(h).squeeze(-1)
        node_weights = segment_softmax(node_scores, batch.node_unit, batch.num_units)
        units = h.new_zeros((batch.num_units, h.shape[-1]))
        units.index_add_(0, batch.node_unit, h * node_weights.unsqueeze(-1))
        units = self.unit_norm(units)
        unit_scores = self.unit_score(units).squeeze(-1)
        unit_weights = segment_softmax(unit_scores, batch.unit_graph, batch.num_graphs)
        graphs = h.new_zeros((batch.num_graphs, h.shape[-1]))
        graphs.index_add_(0, batch.unit_graph, units * unit_weights.unsqueeze(-1))
        return graphs


class RelationNormalizedResidualConv(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.relation_projections = nn.ModuleList(
            (nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in EDGE_TYPES)
        )
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def aggregate(
        self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        (source, target) = edge_index
        aggregated = h.new_zeros(h.shape)
        for relation_id, projection in enumerate(self.relation_projections):
            keep = edge_type == relation_id
            if not keep.any():
                continue
            local_source = source[keep]
            local_target = target[keep]
            messages = projection(h[local_source]).to(h.dtype)
            relation_sum = h.new_zeros(h.shape)
            relation_degree = h.new_zeros(h.shape[0])
            relation_sum.index_add_(0, local_target, messages)
            relation_degree.index_add_(
                0, local_target, torch.ones_like(local_target, dtype=h.dtype)
            )
            aggregated = aggregated + relation_sum / relation_degree.clamp_min(
                1.0
            ).unsqueeze(-1)
        return aggregated

    def forward(
        self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        aggregated = self.aggregate(h, edge_index, edge_type)
        update = F.gelu(self.self_projection(h) + aggregated)
        return self.norm(h + self.dropout(update))


class SCVGraph(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 32768,
        hidden_dim: int = 96,
        layers: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            (RelationNormalizedResidualConv(hidden_dim, dropout) for _ in range(layers))
        )
        self.pool = HierarchicalAttentionPool(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, batch: ProgramGraphBatch) -> dict[str, Any]:
        h = self.input_norm(self.token_embedding(batch.x))
        for layer in self.layers:
            h = layer(h, batch.edge_index, batch.edge_type)
        graph_embedding = self.pool(h, batch)
        logits = self.classifier(graph_embedding)
        return {"logits": logits, "embedding": graph_embedding}


EXAMPLE = 'pragma solidity 0.5.17;\ncontract Vault {\n    mapping(address => uint256) balances;\n    function withdraw(uint256 amount) public {\n        require(block.timestamp > 100);\n        require(balances[msg.sender] >= amount);\n        (bool ok, ) = msg.sender.call.value(amount)("");\n        require(ok);\n        balances[msg.sender] -= amount;\n    }\n}\n'


def build_graph(source, compiler_version="0.5.17"):
    with tempfile.TemporaryDirectory(prefix="obuse_") as directory:
        path = Path(directory) / "Contract.sol"
        path.write_text(source, encoding="utf-8")
        return SCVGraphBuilder().build(path, compiler_version)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--source", type=Path)
    cli.add_argument("--solc", default="0.5.17")
    args = cli.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20260825)
    source = args.source.read_text(encoding="utf-8") if args.source else EXAMPLE
    graph = build_graph(source, args.solc)
    batch = collate_graphs([graph])
    model = SCVGraph()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.0001)
    before = model.classifier.weight.detach().clone()
    logits = model(batch)["logits"]
    loss = F.cross_entropy(logits, torch.tensor([1]), weight=torch.ones(2))
    optimizer.zero_grad()
    loss.backward()
    if not torch.isfinite(loss) or not all(
        (p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    ):
        raise RuntimeError("Non-finite loss or gradient")
    optimizer.step()
    if torch.equal(before, model.classifier.weight):
        raise RuntimeError("Parameters did not update")
    relations = Counter((EDGE_TYPES[index] for index in graph.edge_type))
    if not args.source and (not {"VALUE_FLOW", "CFG_SUMMARY_NEXT"} <= set(relations)):
        raise RuntimeError("Missing graph relations")
    model.eval()
    with torch.no_grad():
        output = model(batch)["logits"]
    print(
        json.dumps(
            {
                "status": "ok",
                "nodes": graph.num_nodes,
                "edges": graph.num_edges,
                "relations": dict(relations),
                "parameters": sum((p.numel() for p in model.parameters())),
                "logits_shape": list(output.shape),
                "loss": float(loss.detach()),
                "optimizer_steps": 1,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
