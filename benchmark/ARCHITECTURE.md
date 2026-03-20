# Benchmark Architecture

## Goal

`benchmark/` 的目标不是只测最终答案，而是把 agent 行为拆成四层，分别做监督和归因：

- `rewriter`
- `router`
- `retrieval`
- `answer`

这样当效果退化时，可以判断问题出在：

- 改写错了
- 工具选错了
- 检索没命中
- 回答阶段幻觉或表述错误

## Code Map

### Entrypoints

- `benchmark/run_benchmark.py`
  - 读取数据集
  - 为每条 case 构建 `AgentWorkflow`
  - 注入 `context`
  - 消费 `workflow.run_stream()`
  - 调 evaluator
  - 聚合 summary

- `benchmark/synthesis/synthesizer.py`
  - 用 LLM 合成 benchmark case
  - Phase 1 先生成输入/对话
  - Phase 2 再标 GT
  - 之后走 validator + quality gate + dedupe

### Contracts

- `benchmark/schema.py`
  - 定义 benchmark dataset 的字段职责
  - 是数据 contract，不是 runtime logic

- `benchmark/config.py`
  - benchmark 根目录、数据目录、结果目录

### Evaluators

- `benchmark/eval/rewriter_eval.py`
- `benchmark/eval/router_eval.py`
- `benchmark/eval/retrieval_eval.py`
- `benchmark/eval/answer_eval.py`
- `benchmark/eval/llm_judge.py`

这些文件只做“给定 GT 与实际行为 -> 产出分数”，不负责数据生成。

### Synthesis Support

- `benchmark/synthesis/models.py`
  - Phase 1 输出草稿的结构约束

- `benchmark/synthesis/validator.py`
  - schema 完整性检查
  - 基本一致性检查

- `benchmark/synthesis/quality.py`
  - 面向 benchmark 价值的质量门
  - 用来拦掉“字段齐了但 case 不干净”的样本

## Runtime Flow

真实 benchmark 运行链路：

1. 读取 `benchmark/data/cases.json`
2. 加载 RAG contexts
3. 为每个 case 创建新的 `AgentWorkflow`
4. 注入 `context`
5. 执行 `workflow.run_stream(case["input"])`
6. 从事件流抽取：
   - `rewrite_result`
   - `actual_calls`
   - `batches`
   - `tool_results`
   - `answer`
7. 只对 `layer_targets` 指定的层打分
8. 写 summary 和 per-case detail

这里的核心依赖是 event contract，而不是内部实现细节。

## Data Contract

每条 case 可以理解成 4 个部分：

- `input`
  - 最终测试轮用户输入

- `context`
  - 测试轮之前注入给 agent 的会话状态
  - 包括：
    - `user_profile`
    - `memory_facts`
    - `conversation_history`

- `dimensions`
  - 切片统计字段
  - 用来做 error analysis
  - 不应该承载 layer GT 本身

- `*_gt`
  - 每层 evaluator 直接消费的监督信号

字段职责要尽量单一，不要混用。

## Router Contract

router 层推荐优先使用 `router_gt.expected_calls`，不要只靠 `expected_tools`。

### Why

`expected_tools` 只能回答：

- “用了哪些工具”

但回答不了：

- “同一个工具要调用几次”
- “每次调用对应哪个车型”
- “每次调用关心哪个参数”

所以现在 router GT 的主表达应该是：

```json
{
  "router_gt": {
    "expected_calls": [
      {"name": "rag_search", "car_model": "EC6", "query_keywords": ["冬季续航损耗"]},
      {"name": "rag_search", "car_model": "EC6", "query_keywords": ["热管理系统"]},
      {"name": "rag_search", "car_model": "ET5T", "query_keywords": ["冬季续航损耗"]},
      {"name": "rag_search", "car_model": "ET5T", "query_keywords": ["热管理系统"]}
    ],
    "expected_tools": ["rag_search"],
    "forbidden_tools": [],
    "no_tool_needed": false,
    "tool_params": {},
    "min_calls": 4,
    "max_calls": 4,
    "must_be_parallel": true
  }
}
```

### Field Roles

- `expected_calls`
  - 主 GT
  - evaluator 按 call 逐条匹配

- `expected_tools`
  - 工具名摘要
  - 兼容旧逻辑与快速统计

- `forbidden_tools`
  - 只禁止“明确不合理”的工具
  - 不要为了绑定某种实现策略而滥禁

- `no_tool_needed`
  - 只有真正不需要任何工具时才为 `true`

- `min_calls` / `max_calls`
  - 应与 `expected_calls` 数量一致，除非你明确允许冗余/压缩

- `must_be_parallel`
  - 只在 case 目标就是“并行发出多条调用”时设为 `true`

## Router Case Design Rules

### 1. Router case should test routing, not domain gaps

如果一个 case 依赖知识库里并不稳定、并不明确存在的字段，那它就不是干净的 router case。

不推荐直接拿这些做 router GT：

- 未稳定收录的车型参数
- 容易受版本或时间影响的价格/政策/新闻
- 数据源里没有明确表述的结论性信息

### 2. Use stable fields first

当前更适合做 router benchmark 的字段：

- `续航`
- `轴距`
- `快充`
- `座椅/座位布局`
- `外观选配`
- `交付周期`
- `换电兼容性`

不够稳定、容易混入知识覆盖问题的字段要谨慎使用，例如：

- `加速性能`
- `功率`
- `电机布局`

这些字段不是永远不能测，但不适合直接作为最基础的 router case 主体。

### 3. Do not over-constrain tool policy

如果某类问题在你当前系统设计里，`rag_search` 和 `grep_search` 都可能合理，就不要把其中一个写成 `forbidden_tools`。

推荐：

- 禁 `web_search`
  - 当问题不是实时信息时

谨慎：

- 禁 `grep_search`
  - 只有当 case 明确在测语义检索型 routing 时才考虑

### 4. Decide call granularity explicitly

同一个 case 要先想清楚测什么：

- 测“多车型并行”
  - 通常按车型拆 call

- 测“多属性拆分”
  - 通常按车型 × 属性拆 call

- 测“单车型多参数聚合效率”
  - 则应故意要求压成 1 个 call

不要一边想测并行，一边又把 GT 写成聚合单 call。

### 5. Keep router cases answer-agnostic

router case 关注的是：

- 是否该调工具
- 调哪个工具
- 调几次
- 参数是否合理

它不应该顺便承担：

- retrieval hit
- answer factual match
- hallucination

这些应该交给 retrieval / answer 层。

## Synthesis Flow

合成分 2 个阶段：

### Phase 1

生成：

- 单句输入
- 或多轮对话草稿

这里主要控制：

- `mode`
- `test_turn_constraint`
- 用户画像、车型、参数、多样性

### Phase 2

给定 Phase 1 草稿，标注：

- `dimensions`
- `context`
- 对应层的 GT block

再经过：

- schema validation
- consistency checks
- quality gates
- duplicate detection

## Current Gaps

当前 benchmark 还不完整，主要缺口在：

- `router` case 数量和规则还在打磨
- `retrieval` 层缺少足够多的 `relevant_chunk_ids`
- `answer` 层还需要更多高质量 GT
- synthesis prompt 仍可能产生“字段齐但 case 不干净”的样本

所以现在最适合的工作方式不是一次性大量合成，而是：

1. 小批量合成
2. 抽检
3. 收紧 quality gate
4. 再扩大规模
