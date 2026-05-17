# 岗位画像与考察重点

## 岗位一览

| 编号 | 岗位 | 核心关键词 |
|------|------|-----------|
| 1 | 大模型算法工程师 | 模型架构、训练范式、分布式训练、算法创新 |
| 2 | 大模型开发工程师 | 工程化落地、服务封装、RAG集成、系统设计 |
| 3 | 大模型数据工程师 | 语料处理、分词策略、数据管道、样本构造 |
| 4 | 推理部署工程师 | 模型量化、推理加速、服务化、多卡部署 |
| 5 | 垂直领域微调工程师 | LoRA/PEFT、指令构造、评估体系、行业适配 |

---

## 各岗位 askbook/RAG 考察深度

| 岗位 | RAG 应用深度 | MCP 协议 | Ingestion | 评估体系 | 自由出题方向 |
|------|-------------|---------|-----------|---------|-------------|
| 1 算法 | 浅（了解流程即可） | 浅 | 浅 | 中 | SFT/RLHF/DPO、分布式训练、Loss 设计 |
| 2 开发 | **深** | **深** | **深** | **深** | API 设计、并发优化、LangChain/LlamaIndex |
| 3 数据 | 中 | 浅 | **深** | 中 | 分词策略、数据清洗、样本构造 |
| 4 部署 | 中 | 中 | 浅 | 中 | 量化/推理加速、TensorRT/vLLM、多卡调度 |
| 5 微调 | 中 | 中 | 中 | **深** | LoRA/PEFT、指令构造、灾难性遗忘 |

**自由出题方向**：interview.md 未覆盖或浅覆盖的领域，面试官需根据自身知识即兴出题。

---

## 1. 大模型算法工程师

**定位：** 负责模型架构设计、训练任务建模、损失函数设计、训练策略制定与效果分析，部分高级岗位参与注意力机制/稀疏结构等算法创新。

**askbook/RAG 考察要点（浅）：**
- 1.1 RAG 的基本定义和工作流程
- 1.7 Bi-Encoder vs Cross-Encoder 区别
- 3.1 Hit Rate / MRR / NDCG 各自含义

**自由出题要点（核心，interview.md 无覆盖）：**
- Transformer 结构细节：Self-Attention、位置编码、FFN、LayerNorm
- 训练范式：预训练（CLM/MLM）-> SFT -> RLHF/DPO 各阶段目标与损失函数
- 分布式训练：数据并行、模型并行、ZeRO、混合精度、梯度累积
- 训练诊断：Loss 曲线分析、梯度爆炸/消失、过拟合判断、调参策略
- 前沿算法：Flash Attention、MoE、LoRA、RoPE、GQA 等

**常用工具：** PyTorch、DeepSpeed、Megatron、ColossalAI、W&B、TensorBoard

**面试题型：** 原理推导、论文复现、Loss 函数设计、训练异常诊断

---

## 2. 大模型开发工程师

**定位：** 连接底层模型与上层应用的工程化执行者，负责模型服务封装、API 设计、RAG 系统集成、多模型管理与系统监控。

**askbook/RAG 考察要点（深，重点考察）：**
- 1.1~1.13 RAG 全链路（Ingestion / chunk_id / ChunkRefiner / 图片检索）
- 2.1~2.6 MCP 协议全套
- 3.1~3.8 评估体系
- 4.1~4.9 Harness + Trace + PII 脱敏
- 5.1~5.7 Python 工程全部
- 6.1~6.7 系统设计全部
- 7.1~7.4 故障排查全部
- 8 露馅警示点

**自由出题要点（辅助）：**
- API 设计：RESTful vs MCP 工具设计、版本管理
- 并发优化：async/await、连接池、请求队列
- LangChain / LlamaIndex 等框架对比

**常用工具：** Python、FastAPI、vLLM、Docker、Redis、K8s

**面试题型：** 系统设计题、代码走查、服务架构设计、故障排查场景题

---

## 3. 大模型数据工程师

**定位：** 负责预训练/微调数据的全生命周期管理：采集、清洗、分词、样本构造、分布式加载、版本管理。

**askbook/RAG 考察要点（中）：**
- 1.3 Chunking 策略（重点）
- 1.10 Ingestion Pipeline 五阶段（重点）
- 1.11 chunk_id 生成与幂等设计
- 1.12 ChunkRefiner / MetadataEnricher
- 1.13 图片检索链路
- 3.5 Golden Dataset 构建

**自由出题要点（核心，interview.md 无覆盖或浅覆盖）：**
- 文本清洗：去重、格式规整、编码处理、质量过滤
- 分词策略：BPE、SentencePiece、WordPiece；词表训练、Token 分布控制
- 样本构造：SFT 指令对、RLHF 偏好对、Few-shot 序列、多轮对话格式
- 数据管道：DataLoader 优化、Sharding、异步加载、流水线设计
- 版本管理：DVC、MLflow、可追溯机制

**常用工具：** Python、Pandas、HuggingFace Datasets、Spark、Ray、Jieba、Airflow

**面试题型：** 数据处理编程题、分词原理、样本构造设计、管道性能优化

---

## 4. 推理部署工程师

**定位：** 将训练好的模型转化为线上高性能推理服务，负责格式转换、量化加速、服务封装、资源调度与监控容错。

**askbook/RAG 考察要点（中）：**
- 1.5 Embedding 选型与维度权衡
- 1.7 Cross-Encoder CPU 延迟与 Graceful Fallback
- 2.1 MCP vs REST
- 2.5 stdio vs HTTP Transport
- 6.3 HNSW 索引原理
- 6.5 100 并发 RAG 服务架构
- 7.2 可靠性设计（重试/熔断/超时）

**自由出题要点（核心，interview.md 无覆盖）：**
- 模型格式转换：PyTorch -> ONNX -> TensorRT；图优化、结构裁剪
- 推理加速：FP16/INT8 量化、KV Cache、Batch 合并、稀疏矩阵
- 多卡部署：张量并行、流水并行、NCCL、显存管理、GPU 负载
- 服务封装：FastAPI/Triton、路由管理、超时处理、日志收集
- 性能分析：Latency/Throughput 测试、瓶颈定位、内存优化

**常用工具：** TensorRT、ONNX Runtime、vLLM、Triton Server、Docker、K8s、Prometheus

**面试题型：** 推理延迟优化设计、量化原理、高并发方案、系统稳定性保障

---

## 5. 垂直领域微调工程师

**定位：** 在预训练模型基础上，针对特定行业/任务进行数据处理、微调策略设计、评估迭代与业务集成。

**askbook/RAG 考察要点（中偏深）：**
- 3.1~3.8 评估体系（**重点**，Hit Rate/MRR/NDCG/RAGAS/LLM-judge）
- 4.1~4.4 Harness 工程（反模式、CI 门禁）
- 4.5 成本追踪与 Token 预算
- 7.3 隔离变量法排查性能退化

**自由出题要点（核心，interview.md 无覆盖）：**
- 微调技术路线：全参数微调 vs LoRA/Adapter/Prefix Tuning；参数量、效果与成本权衡
- Prompt 设计：指令构造、Few-shot/Zero-shot 优化、输出格式控制
- 数据构建：领域语料筛选、指令对设计、边界样本/反事实增强
- 训练稳定性：过拟合诊断、学习率调度、灾难性遗忘防止

**常用工具：** HuggingFace Transformers、PEFT、DeepSpeed、W&B、TensorBoard

**面试题型：** LoRA 原理推导、微调策略选择场景题、评估指标设计、Prompt 优化实战
