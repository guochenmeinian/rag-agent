
1. 跑baseline
```
CONTEXTUAL_RETRIEVAL=false RAG_MILVUS_URI=/Users/xiao/Documents/Github/rag-agent/milvus_baseline.db python benchmark/run_benchmark.py --dataset benchmark/data/cases_rag_critical.json --layers answer --out benchmark/results/exp_baseline_fixed_dataset.json
```

2. baseline + reranker
```
CONTEXTUAL_RETRIEVAL=false RAG_MILVUS_URI=/Users/xiao/Documents/Github/rag-agent/milvus_baseline.db python benchmark/run_benchmark.py --dataset benchmark/data/cases_rag_critical.json --layers answer --out benchmark/results/exp_reranker_only_fixed_dataset.json
```  

3. baseline + contextual retrieval
```
python benchmark/run_benchmark.py --dataset benchmark/data/cases_rag_critical.json --layers answer --out benchmark/results/exp_contextual_fixed_dataset.json
```

4. baseline + reranker + contextual retrieval
```
python benchmark/run_benchmark.py --dataset benchmark/data/cases_rag_critical.json --layers answer --out benchmark/results/exp_reranker_contextual_fixed_dataset.json
```

 - milvus_baseline.db → baseline 用（无 contextual）
 - milvus.db → contextual retrieval 用

 