# Benchmark

## Status

当前 benchmark 有两种运行状态：

- `dry-run` 已确认可跑：会加载数据集、过滤 case、遍历 runner、打印 summary、写结果文件，但不会调用 LLM、工具或 RAG。
- `full-run` 代码路径已接通，但是否能完整跑完取决于本地依赖和密钥：`EXECUTOR_*` / `QWEN_*`、Milvus 本地数据、PDF 解析与 embedding 相关依赖、可选的 `SERPER_API_KEY`。

已在当前仓库确认可执行的命令：

```bash
python benchmark/run_benchmark.py --dry-run
```

## What It Benchmarks

benchmark 不是只看最终 answer，而是把整个 agent pipeline 拆成四层：

- `rewriter`：多轮上下文改写是否独立、实体是否抽对、该澄清时是否澄清
- `router`：工具选型、参数正确性、并行调用是否合理
- `retrieval`：召回结果是否命中 GT chunk，指标为 `hit@k` 和 `MRR`
- `answer`：答案匹配度、幻觉、拒答/澄清是否正确

对应 schema 在 [schema.py](/Users/arist/Documents/learn/projects/rag-agent/benchmark/schema.py)。

## Runtime Pipeline

真实运行时，runner 会按下面的链路执行：

1. 读取 `benchmark/data/cases.json`
2. 为每个蔚来车型加载 `RAGContext`
   - 数据目录优先读 `data/<MODEL>/`
   - 否则回退到 `data/<MODEL>.pdf`
   - collection 名为 `nio_<model.lower()>`
3. 为每个 case 创建新的 `AgentWorkflow`
4. 注入 case 自带 `context`
   - `user_profile -> workflow.memory.global_user_info.raw`
   - `memory_facts -> workflow.memory.facts`
   - `conversation_history -> recent messages`
5. 调用 `workflow.run_stream(case["input"])`
6. 从事件流解析出：
   - `clarify/refined -> rewrite_result`
   - `tool_calling -> actual_calls + batches`
   - `tool_done -> tool_results`
   - `done -> answer`
7. 按 `layer_targets` 决定实际打哪些分
8. 聚合 summary，并输出到 `benchmark/data/results/run_<timestamp>.json`

## Event Contract

runner 依赖 [workflow.py](/Users/arist/Documents/learn/projects/rag-agent/src/agent/workflow.py) 的事件流，关键事件是：

- `{"type": "clarify", "message": str}`
- `{"type": "refined", "query": str}`
- `{"type": "tool_calling", "calls": [{"name": str, ...}]}`
- `{"type": "tool_done", "results": list[dict]}`
- `{"type": "done", "answer": str, "tool_results": list[dict] | None}`

这意味着 benchmark 实际测的是完整 agent 行为，而不是离线字符串对比。

## Dataset Shape

每条 case 的核心字段：

- `id`
- `layer_targets`
- `category`
- `dimensions`
- `context`
- `input`
- `rewriter_gt | router_gt | retrieval_gt | answer_gt`

`context` 用来模拟“当前对话不是第一轮”，`dimensions` 用来做切片统计，比如：

- `is_multi_turn`
- `history_length`
- `has_coref`
- `depends_on_memory`
- `is_ambiguous`
- `should_clarify`

当前仓库内的样例数据现状：

- `benchmark/data/cases.json` 目前只有 `18` 条 case
- 全部都是 `rewriter` 层 case
- category 目前只有：
  - `rw_standalone_coref`
  - `rw_standalone_ellipsis`
  - `rw_standalone_topic_shift`
  - `rw_entity_extraction`
  - `rw_clarify_ambiguous`
  - `rw_clarify_safety`

所以现在仓库内自带的是一个小样本，不是完整 benchmark 资产。

## Metrics

### Rewriter

在 [eval/rewriter_eval.py](/Users/arist/Documents/learn/projects/rag-agent/benchmark/eval/rewriter_eval.py)：

- `standalone`
  - hard check：`coref_map` 是否被正确解析、`ellipsis_slots` 是否填充
  - soft check：LLM judge 比较 rewrite 是否真正脱离上下文
- `entity_extraction_accuracy`
  - hard check：`required_entities` 必须出现，`forbidden_entities` 不得出现
  - soft check：LLM judge 看术语/实体是否合理
- `clarify_detection`
  - hard check：该澄清时必须输出 `clarify`

### Router

在 [eval/router_eval.py](/Users/arist/Documents/learn/projects/rag-agent/benchmark/eval/router_eval.py)：

- `tool_classification_accuracy`
  - `expected_tools` 必须被调用
  - `forbidden_tools` 不能被调用
  - `no_tool_needed=true` 时不能乱调工具
- `parameter`
  - `correct_tool`
  - `correct_format`
  - `correct_content`
- `multi_query`
  - `efficient`：总调用数不能超过 `max_calls`
  - `complete`：调用数不能少于 `min_calls`
  - `parallel_ok`：`must_be_parallel=true` 时，预期工具必须在同一 batch 中发出

### Retrieval

在 [eval/retrieval_eval.py](/Users/arist/Documents/learn/projects/rag-agent/benchmark/eval/retrieval_eval.py)：

- `hit@k`
- `MRR`

前提是 case 里提供了 `retrieval_gt.relevant_chunk_ids`。如果没有这个字段或为空，会被标记为 `skip`。

### Answer

在 [eval/answer_eval.py](/Users/arist/Documents/learn/projects/rag-agent/benchmark/eval/answer_eval.py)：

- `match`
  - hard check：`key_facts` 是否出现
  - soft check：LLM judge 给 `0/1/2`
- `hallucination`
  - hard check：`forbidden_content` 不得出现
  - soft check：LLM judge 判断是否编造
- `clarification`
  - LLM judge 判断是否正确表达不确定性或拒绝

## Synthesis Pipeline

合成器在 [synthesis/synthesizer.py](/Users/arist/Documents/learn/projects/rag-agent/benchmark/synthesis/synthesizer.py)，是两阶段流程。

### Phase 1: 输入/对话生成

按 target 的 `mode` 分三种：

- `single`
  - 直接生成一句用户输入
  - 适合 router / retrieval / 大部分 answer
- `short`
  - 生成 2-3 轮短对话
  - 最后一条 user message 是 test turn
  - 适合 coref / ellipsis / topic shift / ambiguous clarify
- `long_memory`
  - 生成 6-8 轮对话
  - 第 1-2 轮植入早期信息
  - 中间插漂移内容
  - 最后一问依赖早期 memory

当前参数：

- `Phase 1 temperature = 1.05`
- `Phase 1 presence_penalty = 0.3`
- `Phase 1 max_tokens`
  - `single = 400`
  - `short = 1400`
  - `long_memory = 2400`

Phase 1 会采样：

- `15` 个用户画像 `USER_PROFILES`
- `8` 个车型 `ALL_CARS`
- `12` 类参数种子 `PARAM_SEEDS`
- `4` 种对话风格 `CONV_STYLES`

并用 `DiversityTracker` 降低重复采样概率。

### Phase 1.5: 结构校验和质量门

这是当前新增的部分：

- [models.py](/Users/arist/Documents/learn/projects/rag-agent/benchmark/synthesis/models.py)
  - `SingleTurnDraft`
  - `ConversationDraft`
- [quality.py](/Users/arist/Documents/learn/projects/rag-agent/benchmark/synthesis/quality.py)

质量门目前会拦掉这些 case：

- 输入太短，几乎没有 benchmark 价值
- `single` 模式却带了历史
- `short/long_memory` 模式却没有历史
- `long_memory` 历史过短，不像真的记忆题
- `has_coref=true` 但测试轮没有明显指代表达
- `rw_clarify_safety` 没有明显 injection / harmful intent 痕迹

### Phase 2: GT 标注

给定 Phase 1 的草稿，再调用一次 LLM 标注 GT。当前参数：

- `temperature = 0.2`
- `max_tokens = 1500`

Phase 2 输出：

- 顶层 case 元数据
- `dimensions`
- `context`
- 对应 layer 的 GT block

之后还会过：

- schema validation
- consistency checks
- duplicate detection

重复检测目前用字符 4-gram Jaccard，阈值 `0.75`。

## Targets And Intended Scale

synthesizer 里定义了 `21` 个 target，覆盖四层 benchmark，目标规模是：

- `21 targets × 18 cases = 378 cases`

但当前仓库实际提交的数据还远没到这个规模，所以 README 里的这个数字是“设计目标”，不是“当前已有数据量”。

## Commands

运行 benchmark：

```bash
python benchmark/run_benchmark.py
python benchmark/run_benchmark.py --layers rewriter router
python benchmark/run_benchmark.py --ids rw_standalone_coref_001,rw_entity_001
python benchmark/run_benchmark.py --category rw_standalone_coref
python benchmark/run_benchmark.py --dataset benchmark/data/cases.json --out benchmark/data/results/run_01.json
python benchmark/run_benchmark.py --dry-run
```

生成数据：

```bash
python benchmark/synthesis/synthesizer.py --target rw_standalone_coref rt_parameter --n 5
python benchmark/synthesis/synthesizer.py --n 18 --out benchmark/data/cases.json
python benchmark/synthesis/synthesizer.py --append --out benchmark/data/cases.json
```

## Practical Requirements

想跑 `full-run`，至少要满足：

- Executor 模型可调用：`EXECUTOR_MODEL / EXECUTOR_API_KEY / EXECUTOR_BASE_URL`
- Rewriter 模型可调用：`QWEN_MODEL / QWEN_API_KEY / QWEN_BASE_URL`
- 本地 `data/` 下存在 PDF 或已建立可复用的 Milvus collection
- `milvus.db` 和 `grep_index.db` 路径可写
- 若 case 触发 `web_search`，需要 `SERPER_API_KEY`

## Known Limits

- 当前自带数据只有 rewriter 样本，router / retrieval / answer 还没有完整 case 集
- `retrieval` 的 GT 依赖稳定 chunk id；只要 chunking 变化，`relevant_chunk_ids` 就可能漂移
- `answer` 和 `rewriter` 的 soft metric 依赖 LLM judge，适合迭代，不适合当唯一真值
- 现在还没有显式的 `difficulty` / `failure_modes` 字段，所以“为什么挂”仍然不够可解释

## Recommended Next Step

最值得继续做的不是先堆更多 case，而是补两层数据控制：

- 给 case 增加 `difficulty`
- 给 case 增加 `failure_modes`

这样 benchmark 就能从“有分数”升级成“能解释系统在哪类 hard case 上挂掉”。
