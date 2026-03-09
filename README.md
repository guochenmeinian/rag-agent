# RAG-Agent: 蔚来汽车智能问答

基于 RAG + Agent 的蔚来汽车购车顾问，支持多轮对话、工具调用、记忆与反思。

## 特性

- **RAG 检索**：LlamaParse 解析 PDF → BGE-M3 嵌入 → Milvus 混合检索
- **Agent 流程**：Rewrite（查询改写）→ Execute（工具调用/直接回答）→ Reflect（答案校验）
- **多轮记忆**：fact list + 最近消息，支持指代消解与上下文追问
- **可选模型**：Executor 可切换 OpenAI / Kimi / DeepSeek 等；Qwen 系负责改写与反思

## 目录结构

```
rag-agent/
├── src/
│   ├── config.py          # 配置（含 get_executor_cfg / get_qwen_cfg）
│   ├── main.py            # REPL 入口
│   ├── api.py             # FastAPI 入口
│   ├── frontend.py        # Streamlit 入口
│   ├── cache_cli.py      #  ingest 缓存管理
│   ├── agent/             # Rewriter / Executor / Reflector / Memory
│   ├── prompts/           # 各组件的 system prompt
│   ├── rag/               # 解析、分块、嵌入、检索
│   ├── storage/           # Milvus + ingest 管理
│   ├── tools/             # rag_search / web_search
│   └── tests/             # 评估与 benchmark
├── data/                  # 蔚来车型 PDF（EC6.pdf, ET5.pdf 等）
├── .env                   # API key、模型配置
└── run.sh                 # setup / serve
```

## 安装与运行

```bash
# 1. 安装
./run.sh setup

# 2. 配置 .env（复制 .env.example 并填写）
cp .env.example .env

# 3. 启动（三选一）
./run.sh serve                              # FastAPI → http://localhost:8000
cd src && streamlit run frontend.py         # Streamlit
PYTHONPATH=src .venv/bin/python src/main.py --session demo   # 命令行 REPL
```

## 配置：模型选择

在 `.env` 中配置各组件使用的模型，可分开指定 Executor 与 Qwen 系：

| 组件 |  env 变量 | 默认 | 说明 |
|------|-----------|------|------|
| **Executor**（推理引擎） | `EXECUTOR_MODEL`<br>`EXECUTOR_API_KEY`<br>`EXECUTOR_BASE_URL` | gpt-4o / OpenAI | 工具调用与回答，可换成 Kimi / DeepSeek |
| **Qwen 系**（改写、反思、记忆） | `QWEN_MODEL`<br>`QWEN_API_KEY`<br>`QWEN_BASE_URL` | qwen3.5-instruct / DashScope | 多轮改写与答案校验 |

**使用 Kimi 作为 Executor：**

```env
EXECUTOR_MODEL=moonshot-v1-8k
EXECUTOR_API_KEY=sk-...
EXECUTOR_BASE_URL=https://api.moonshot.cn/v1
```

**使用 DeepSeek：**

```env
EXECUTOR_MODEL=deepseek-chat
EXECUTOR_API_KEY=sk-...
EXECUTOR_BASE_URL=https://api.deepseek.com/v1
```

兼容旧变量：`OPENAI_MODEL` / `OPENAI_API_KEY` 会作为 Executor 的 fallback；`DASHSCOPE_API_KEY` 会作为 Qwen 的 fallback。

## Agent 流程

```
用户输入
  → Memory.add_message(user)
  → format_for_prompt() 生成 [用户背景][用户记忆][最近消息]
  → Rewriter 改写为 standalone 问题
  → Executor 决策：tool_call 或 direct
       ├─ tool_call → rag_search / web_search（并行）
       └─ direct    → Reflector 校验 grounding
                        ├─ 通过 → update_facts, persist, done
                        └─ 未过 → 带反馈重试
```

## 后续优化方向

- 多轮 benchmark 自动化
- 抽取质量评估与 prompt 微调
- 可选：用 Qwen 等替代 Executor 降低成本

---

## 参考

- 早期实现：`v1.ipynb`、`v1.pdf`（Agent+RAG 检索实现）
- [LlamaCloud Demo](https://github.com/run-llama/llamacloud-demo)
- [Milvus + BGE-M3](https://milvus.io/docs/embed-with-bgm-m3.md)
