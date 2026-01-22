我的第一个想法是 使用强化学习量子架构搜索VQE线路（包括HEA以及UCC线路），以实现更大体系的精确计算。目标是得到比HEA_Rylinear、UCCSD or adaptvqe更浅的线路。
其中搜索空间的池子可以为硬件高效的门（Rx、Ry、Rz and CNOT or SWAP），但这部分工作已经有文献实现了（https://arxiv.org/pdf/2103.16089），后续我也会在literature文件夹中创建一个属于RLQAS的文献库。
但对于UCC线路来说是创新的，我们可以将算符池定义为UCC的单双（三）SD(T)激发算符、或者是李震宇提出的sQEB算符以适应硬件（https://pubs.acs.org/doi/10.1021/acs.jctc.5c00119）
通过RLQAS来搜索线路，目标是尽可能模拟大的体系例如（20qubits左右）

第二个做法是这样的：将我的QAS文献库导入到literature文件夹中，或者通过agent-skills搜索更多的关于QAS或者量子线路设计的文章，形成一个文献库。通过Agent or LLM阅读这些文献，以及我和他们的chat。
让不同的agent和LLM提出一些关于可以解决线路设计中线路深度与表达能力平衡的问题。（例如提出如何设计一个更高效的激发算符的线路，设计算符如何排列组合算符等可能的解决办法）
Anyway，总之就是和Agent聊一些可能的idea出来，然后最后通过“牛马”工具例如Ralph、code-agent等来实现，最好是可以解决一些存在的科学问题。
