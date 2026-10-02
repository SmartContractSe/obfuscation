# obuse

SCVGraph `valueflow_only`：Solidity 源码 → 语义节点、CFG 路径摘要与精确值流 → 六层关系图网络 → 二分类。代码独立于原工程。

Python 3.10+，在本目录运行：

```bash
python3 -m pip install torch==2.11.0 solidity-parser==0.0.7 slither-analyzer==0.11.5 solc-select==1.2.0
solc-select install 0.5.17
python3 obuse.py
```

内置一个合约，完成构图、前向计算、反向传播和一次参数更新。实测输出 `status: ok`，19 个节点、94 条边、3,986,884 个参数，输出维度 `[1, 2]`。

自有合约：`python3 obuse.py --source contract.sol --solc 0.5.17`，编译器版本须匹配并已安装。模型为随机初始化，入口的标签仅用于验证训练步骤；检测应用需自行训练。可直接导入 `build_graph`、`collate_graphs` 和 `SCVGraph`。
