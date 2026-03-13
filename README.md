# RAG-Agent：NIO 智能问答

基于 RAG + Agent 的 NIO 购车顾问，支持多轮对话、并行工具调用与对话记忆。

## 架构概览

```
用户输入
  → Memory.format_for_prompt()          [用户背景][用户记忆][最近N条消息]
  → Qwen Rewriter
      ├─ clarify / reject  →  直接返回澄清/拒绝消息，不进 Executor
      └─ rewrite           →  refined standalone query
  → GPT-4o Executor（context_prompt 注入 system message）
      ├─ tool_call  →  ToolRegistry.run_parallel()  →  回到 Executor
      │                   ├─ rag_search   Milvus 混合检索
      │                   ├─ grep_search  SQLite 全文检索
      │                   └─ web_search   Serper 网络搜索
      └─ direct     →  Memory.update_facts() + persist  →  done
```

### 关键设计点

- **Rewriter 两路分支**：改写走正常流程；检测到超范围/18禁/Prompt 注入时直接 clarify，不进 Executor，节省一次大模型调用
- **并行工具调用**：Executor 在一次响应中发出所有 tool call（多车型对比、NIO + 竞品），Registry 用 `ThreadPoolExecutor` 并发执行，带超时和指数退避重试
- **两种检索互补**：`rag_search` 做语义检索适合概念/推荐；`grep_search` 做精确关键词匹配适合轴距/续航等具体参数
- **context_prompt 注入 system**：记忆上下文追加到 Executor system message 而非 user message，避免与当前问题混用
- **max iterations 保底**：若全部轮次都是 tool call，用 `force_direct=True` 再调一次 Executor 强制产出直接回答

## 目录结构

```
rag-agent/
├── src/
│   ├── config.py               配置（API key、模型、路径均可 .env 覆盖）
│   ├── main.py                 命令行 REPL
│   ├── api.py                  FastAPI 服务
│   ├── frontend.py             Streamlit UI（旧版实现，仍可运行）
│   ├── cache_cli.py            ingest 缓存管理工具
│   ├── agent/
│   │   ├── workflow.py         主编排器（run / run_stream）
│   │   ├── planner.py          QueryRewriter（Qwen）
│   │   ├── executor.py         AgentExecutor（GPT-4o / 可换）
│   │   ├── memory.py           ConversationMemory：facts + 滑动窗口
│   │   └── state.py            AgentState 数据类
│   ├── prompts/
│   │   ├── rewriter.py         Rewriter system prompt（改写规则 + few-shot）
│   │   ├── executor.py         Executor system prompt（工具决策 + 并行规则）
│   │   └── memory.py           fact 提取 prompt
│   ├── rag/
│   │   ├── pipeline.py         ingest + retrieve 入口
│   │   ├── embedder.py         BGE-M3 嵌入（dense + sparse）
│   │   ├── retriever.py        Milvus hybrid_search
│   │   ├── chunker.py          文本分块
│   │   └── parser.py           LlamaParse PDF 解析
│   ├── storage/
│   │   ├── vector_store.py     MilvusVectorStore
│   │   ├── grep_index.py       GrepIndex（SQLite FTS）
│   │   └── ingest_manager.py   文件指纹 + parse 缓存
│   ├── tools/
│   │   ├── base.py             BaseTool（Pydantic 验证 + OpenAI schema 自动生成）
│   │   ├── registry.py         ToolRegistry（并行执行、超时、重试）
│   │   ├── rag_search.py       RagSearchTool
│   │   ├── grep_search.py      GrepSearchTool
│   │   ├── web_search.py       WebSearchTool（Serper）
│   │   └── result.py           ToolResult
│   └── tests/
│       ├── eval_runner.py      端到端评估（tool recall/precision/keyword hit）
│       └── benchmark_tool_selection.py
├── frontend/                   React + Vite 前端
├── data/                       NIO 车型 PDF（EC6.pdf, ET5.pdf …）
├── .env                        API key 与模型配置
└── run.sh                      setup / serve 脚本
```

## 安装与运行

```bash
# 1. 安装依赖
./run.sh setup

# 2. 配置（复制模板并填写 key）
cp .env.example .env

# 3. 启动
./run.sh serve                                              # 后端 FastAPI（uvicorn）→ http://localhost:8000
cd frontend && npm install && npm run dev                  # 前端 React + Vite
PYTHONPATH=src .venv/bin/python src/main.py --session demo  # 命令行 REPL

# 旧版前端（可选）
cd src && streamlit run frontend.py
```

## 模型配置

两套模型可独立配置：

| 组件 | 环境变量 | 默认 | 说明 |
|------|----------|------|------|
| **Executor**（推理 + 工具调用） | `EXECUTOR_MODEL`<br>`EXECUTOR_API_KEY`<br>`EXECUTOR_BASE_URL` | gpt-4o / OpenAI | 可替换为 Kimi、DeepSeek 等兼容 OpenAI API 的模型 |
| **Qwen 系**（改写 + 记忆提取） | `QWEN_MODEL`<br>`QWEN_API_KEY`<br>`QWEN_BASE_URL` | qwen3.5-instruct / DashScope | 负责 query rewrite 和 fact 提取 |

兼容旧变量：`OPENAI_MODEL` / `OPENAI_API_KEY` 作为 Executor fallback；`DASHSCOPE_API_KEY` 作为 Qwen fallback。

**示例：切换 Executor 为 DeepSeek**

```env
EXECUTOR_MODEL=deepseek-chat
EXECUTOR_API_KEY=sk-...
EXECUTOR_BASE_URL=https://api.deepseek.com/v1
```

## 记忆机制

每轮对话结束后，Qwen 从对话中提取原子事实（`facts`）并更新列表，同时保留最近 N 条原文消息（`MAX_RECENT_MESSAGES`，默认 5）。下一轮时序列化为：

```
[用户背景]   预算45万，两孩家庭，关注ES8
[用户记忆]
- 用户已看过 ES8 和 ET7
- 对快充时间比较在意
[最近5条消息]
  user: ES8 续航怎么样？
  assistant: ...
```

注入 Executor system message，支持指代消解、省略补全、跨轮追踪。

## 评估

```bash
cd src && python -m tests.eval_runner
# 可选过滤
python -m tests.eval_runner --category multi_car_compare
python -m tests.eval_runner --ids q001,q004 --out results.json
```

指标：`tool_recall` / `tool_precision` / `parallel_ok` / `keyword_hit` / `hallucination_hits`

## 参考

- [LlamaCloud Demo](https://github.com/run-llama/llamacloud-demo)
- [Milvus + BGE-M3 Hybrid Search](https://milvus.io/docs/embed-with-bgm-m3.md)
