# obuse

SCVGraph `valueflow_only`: Solidity source code → semantic nodes, CFG path summaries, and precise value-flow relations → a six-layer relational graph network → binary classification. The implementation is independent of the original project.

Python 3.10+ is required. Run the following commands in this directory:

```
python3 -m pip install torch==2.11.0 solidity-parser==0.0.7 slither-analyzer==0.11.5 solc-select==1.2.0
solc-select install 0.5.17
python3 obuse.py
```

The built-in example contains one contract and performs graph construction, forward propagation, backpropagation, and one parameter update. In our test, it produces `status: ok`, with 19 nodes, 94 edges, 3,986,884 parameters, and an output shape of `[1, 2]`.

To test a custom contract, run:

```
python3 obuse.py --source contract.sol --solc 0.5.17
```

The specified compiler version must match the contract and must already be installed. The model is randomly initialized, and the label provided at the entry point is used only to verify the training procedure. For actual vulnerability detection, the model must be trained separately.

The `build_graph`, `collate_graphs`, and `SCVGraph` components can also be imported directly.
