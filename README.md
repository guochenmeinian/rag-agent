# RAG-Agent：NIO 智能问答

基于 RAG + Agent 架构的蔚来汽车购车顾问，支持多轮对话、并行工具调用与跨会话记忆。

---

## 系统设计

### 整体架构

```
用户输入
  → [Qwen] 意图判断 & Query 改写
      ├─ clarify  →  直接返回追问，不进检索
      └─ rewrite  →  精炼为独立查询
  → [GPT-4o] 推理 + Tool Use 循环
      ├─ tool_call  →  并行执行工具  →  注入结果  →  继续循环
      │                ├─ rag_search   向量混合检索（语义）
      │                ├─ grep_search  全文关键词检索（精确）
      │                └─ web_search   联网搜索
      └─ direct    →  更新记忆，返回最终答案
```

### 双模型分工

系统使用两套模型，各司其职：

- **Qwen（轻量）**：负责 Query 改写、意图判断、对话事实提取。调用频繁，用成本低的小模型。
- **GPT-4o（推理）**：负责工具调用决策和最终答案生成。只在必要时调用，支持替换为任意兼容 OpenAI API 的模型（DeepSeek、Kimi 等）。

### 检索设计

针对汽车手册场景，检索层做了两项核心优化：

**1. 双路混合检索**
同时运行 Dense（语义向量，BGE-M3）和 Sparse（BM25 关键词）两路检索，再由 Cross-Encoder Reranker 精排。语义检索擅长概念/推荐类问题，BM25 擅长精确数字/参数名匹配，两者互补。

**2. Table-to-Text 分块**
车型手册大量内容以 Markdown 表格形式存在，直接 embedding 效果差。Chunker 会把表格每一行转为自然语言句子（如"ET5续航：CLTC为660km"），作为 embedding 输入；但给 LLM 看的始终是原始完整表格（Parent Chunk）。这样既保证检索准确，又保证 LLM 上下文完整。

**3. Contextual Retrieval**
Ingest 时对每个 chunk 用 Qwen 生成 1-2 句上下文描述（所属车型、参数类别）并拼接到 chunk 前，帮助向量模型理解 chunk 在文档中的位置。

### 对话记忆

每轮对话结束后，Qwen 从当前轮提取原子事实并维护一个事实列表：

```
[用户背景]   预算 45 万，两孩家庭，关注 ES8
[用户记忆]
- 用户已对比过 ES8 和 ET7
- 对快充时间比较在意
[最近 5 条消息]
  user: ES8 续航怎么样？
  ...
```

这段上下文注入 Executor 的 system message，支持多轮对话中的指代消解和省略补全。会话状态序列化为 JSON 持久化，重启后可继续上次对话。

### 工具并行与容错

LLM 可在一次响应中同时发起多个工具调用（如同时查多个车型），ToolRegistry 用线程池并发执行，每个工具独立设置超时和指数退避重试，单个工具失败不影响其他工具。

---

## 目录结构

```
rag-agent/
├── src/
│   ├── agent/          对话编排（workflow、memory、rewriter、executor）
│   ├── rag/            文档处理（parse、chunk、embed、retrieve）
│   ├── storage/        向量库（Milvus）与全文索引（SQLite FTS）
│   ├── tools/          工具层（rag_search、grep_search、web_search）
│   ├── prompts/        各组件 system prompt
│   ├── api.py          FastAPI 后端
│   └── main.py         命令行 REPL
├── frontend/           React + Vite 前端
├── benchmark/          评估框架 + 第三方系统对比
└── data/               NIO 车型 PDF
```

---

## 安装与运行

```bash
# 1. 安装依赖
./run.sh setup

# 2. 配置
cp .env.example .env   # 填写 API Key

# 3. 启动
./run.sh serve                                              # 后端 → http://localhost:8000
cd frontend && npm install && npm run dev                  # 前端
PYTHONPATH=src python src/main.py --session demo           # 命令行
```

## 模型配置

| 组件 | 环境变量 | 默认 |
|------|----------|------|
| Executor（推理 + 工具调用） | `EXECUTOR_MODEL` / `EXECUTOR_API_KEY` / `EXECUTOR_BASE_URL` | gpt-4o |
| Qwen 系（改写 + 记忆提取） | `QWEN_MODEL` / `QWEN_API_KEY` / `QWEN_BASE_URL` | qwen3.5-flash / DashScope |

---

## Benchmark：与主流 RAG 系统对比

使用相同的 10 个问答用例和 8 份 NIO 车型 PDF，与 5 个主流 RAG 系统横向对比，由 GPT-4o 统一评分。

| 系统 | 技术栈 | 答案准确性 ↑ | 无幻觉率 ↑ | 关键事实覆盖 ↑ |
|------|--------|:-----------:|:----------:|:--------------:|
| **本系统** | BGE-M3 混合检索 + GPT-4o Agent | **1.10** | **1.00** | **0.672** |
| R2R | pgvector 混合检索 + GPT-4o | **1.10** | 0.60 | 0.657 |
| Dify | Weaviate 向量检索 + GPT-4o | 0.80 | 0.70 | 0.582 |
| RAGFlow | Elasticsearch + DeepDoc + GPT-4o | 0.60 | 0.60 | 0.421 |
| AnythingLLM | LanceDB 向量检索 + GPT-4o | 0.20 | 0.10 | 0.253 |
| LightRAG | 知识图谱 + NanoVectorDB + GPT-4o | 0.20 | 0.50 | 0.205 |

**结论**：本系统在答案准确性上与 R2R 并列第一，幻觉率为零（唯一）。Table-to-Text 分块和双路混合检索是数值类问题（续航/参数）精准召回的关键。

---

## 参考

- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Milvus BGE-M3 Hybrid Search](https://milvus.io/docs/embed-with-bgm-m3.md)
- [LlamaCloud Demo](https://github.com/run-llama/llamacloud-demo)
