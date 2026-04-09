"""
检索路由模块：根据用户问题和已选表的DDL，判断是否需要进行值检索和知识库检索
使用 DeepSeek Tool Calls 进行工具调用决策
"""
import json
import time
import threading
from typing import Dict, List, Optional, Tuple
from openai import OpenAI

import config

LOG_FILE = "./frontend.log"
_log_lock = threading.Lock()

def _ts():
    return time.strftime("%H:%M:%S.", time.localtime()) + f"{time.time() % 1:.3f}"[2:]

def _log(msg):
    line = f"[{_ts()}] [RETRIEVAL] {msg}"
    with _log_lock:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
            f.flush()
    print(line, flush=True)

# 延迟导入 heavy dependencies
_faiss = None
_np = None
_AutoTokenizer = None
_BM25Okapi = None

def _import_heavy_deps():
    global _faiss, _np, _AutoTokenizer, _BM25Okapi
    if _faiss is not None:
        return
    t0 = time.time()
    _log("Loading heavy dependencies (faiss, numpy, transformers, rank_bm25)...")
    try:
        import faiss as _faiss_mod
        import numpy as _np_mod
        from transformers import AutoTokenizer as _AutoTokenizer_mod
        try:
            from rank_bm25 import BM25Okapi as _BM25Okapi_mod
        except ImportError:
            raise ImportError("Please install rank_bm25: pip install rank_bm25")
        _faiss = _faiss_mod
        _np = _np_mod
        _AutoTokenizer = _AutoTokenizer_mod
        _BM25Okapi = _BM25Okapi_mod
        _log(f"Heavy dependencies loaded successfully in {(time.time()-t0)*1000:.1f}ms")
    except ImportError as e:
        if "CXXABI" in str(e) or "libstdc++" in str(e):
            _log(f"WARNING: faiss cannot be loaded due to system libstdc++ version mismatch.")
            _log(f"  Error: {e}")
            _log(f"  Retrieval will be skipped. To fix: install a newer libstdc++ or use conda to install faiss-cpu.")
            _faiss = "MISSING"  # Sentinel to indicate faiss unavailable
            _log("Heavy dependencies load completed (faiss unavailable)")
        else:
            raise


class DatabaseSearcher:
    """混合检索器：Faiss向量检索 + BM25稀疏检索，RRF融合"""
    
    def __init__(self, tokenizer_path: str, indices_path: str, metadata_path: str):
        _import_heavy_deps()
        if _faiss == "MISSING":
            raise RuntimeError("faiss is unavailable due to system libstdc++ version mismatch. Retrieval skipped.")
        import pickle
        
        t0 = time.time()
        _log(f"DatabaseSearcher init: loading tokenizer from {tokenizer_path}")
        self.tokenizer = _AutoTokenizer.from_pretrained(tokenizer_path)
        _log(f"DatabaseSearcher init: tokenizer loaded in {(time.time()-t0)*1000:.1f}ms")
        
        t1 = time.time()
        _log(f"DatabaseSearcher init: loading faiss indices from {indices_path}")
        with open(indices_path, "rb") as f:
            serialized_indices = pickle.load(f)
        self.indices = {key: _faiss.deserialize_index(s_index) for key, s_index in serialized_indices.items()}
        _log(f"DatabaseSearcher init: faiss indices loaded in {(time.time()-t1)*1000:.1f}ms ({len(self.indices)} columns)")

        t2 = time.time()
        _log(f"DatabaseSearcher init: loading metadata from {metadata_path}")
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        _log(f"DatabaseSearcher init: metadata loaded in {(time.time()-t2)*1000:.1f}ms ({len(self.metadata)} entries)")
        
        self.client = OpenAI(
            base_url=config.MODELSCOPE_BASE_URL,
            api_key=config.MODELSCOPE_API_KEY,
        )
        self.embed_model_name = 'Qwen/Qwen3-Embedding-0.6B'

    def _embed_queries_api(self, queries: list):
        t0 = time.time()
        _log(f"Embedding {len(queries)} queries via API (model={self.embed_model_name})")
        try:
            response = self.client.embeddings.create(
                model=self.embed_model_name,
                input=queries,
                encoding_format="float"
            )
            embeddings = [item.embedding for item in response.data]
            elapsed = (time.time() - t0) * 1000
            _log(f"Embedding completed in {elapsed:.1f}ms, shape=({len(embeddings)}, {len(embeddings[0]) if embeddings else 0})")
            return _np.array(embeddings, dtype=_np.float32)
        except Exception as e:
            _log(f"Embedding FAILED in {(time.time()-t0)*1000:.1f}ms: {e}")
            raise e

    def search(self, queries: list, top_k: int = 3, threshold: float = 0.5, target_tables: list = None, mode: str = "value", rag_columns: list = None) -> str:
        _log(f"SEARCH START: mode={mode}, queries={queries}, top_k={top_k}, threshold={threshold}, target_tables={target_tables}")
        t_start = time.time()
        
        if mode == "RAG" and not rag_columns:
            raise ValueError("RAG mode requires rag_columns parameter")

        t_emb = time.time()
        query_embeddings = self._embed_queries_api(queries)
        _log(f"Embedding phase took {(time.time()-t_emb)*1000:.1f}ms")
        
        collected_results = {}
        target_tables_lower = [t.lower() for t in target_tables] if target_tables else None

        t_search = time.time()
        columns_searched = 0
        columns_with_results = 0
        
        for index_key, index in self.indices.items():
            table, column = index_key.split('__')
            if target_tables_lower and table.lower() not in target_tables_lower:
                continue

            column_values = self.metadata[index_key]["values"]
            bm25_model = self.metadata[index_key]["bm25"]
            columns_searched += 1
            
            search_pool_size = top_k * 2
            distances, indices_matrix = index.search(query_embeddings, search_pool_size)

            value_scores = {}
            for qi, q in enumerate(queries):
                faiss_ranks = {idx: r + 1 for r, idx in enumerate(indices_matrix[qi]) if idx != -1}

                tokenized_q = self.tokenizer.encode(q, add_special_tokens=False)
                bm25_scores = bm25_model.get_scores(tokenized_q)
                bm25_top_indices = _np.argsort(bm25_scores)[::-1][:search_pool_size]
                bm25_ranks = {idx: r + 1 for r, idx in enumerate(bm25_top_indices) if bm25_scores[idx] > 0}

                combined_indices = set(faiss_ranks.keys()) | set(bm25_ranks.keys())
                for idx in combined_indices:
                    value = column_values[idx]
                    if mode == "value" and len(value) >= 100:
                        continue

                    r_dense = faiss_ranks.get(idx, 1000)
                    r_sparse = bm25_ranks.get(idx, 1000)
                    rrf_score = 30.5 * (1.0 / (60 + r_dense) + 1.0 / (60 + r_sparse))

                    if threshold and rrf_score < threshold:
                        continue

                    if value not in value_scores or rrf_score > value_scores[value]:
                        value_scores[value] = rrf_score
            
            if value_scores:
                columns_with_results += 1
                if table not in collected_results:
                    collected_results[table] = {}
                sorted_values = sorted(value_scores.items(), key=lambda item: item[1], reverse=True)
                collected_results[table][column] = sorted_values[:top_k]
        
        _log(f"Search phase: {columns_searched} columns searched, {columns_with_results} columns with results, took {(time.time()-t_search)*1000:.1f}ms")

        if not collected_results:
            _log(f"SEARCH END: NULL (no results found), total {(time.time()-t_start)*1000:.1f}ms")
            return "NULL"

        if mode == "RAG":
            rag_final_items = []
            for table, columns_data in collected_results.items():
                for column, values_with_scores in columns_data.items():
                    if column in rag_columns:
                        for val, score in values_with_scores:
                            rag_final_items.append((val, score))
            if not rag_final_items:
                _log(f"SEARCH END: NULL (no RAG items), total {(time.time()-t_start)*1000:.1f}ms")
                return "NULL"

            unique_items = {}
            for val, score in rag_final_items:
                if val not in unique_items or score > unique_items[val]:
                    unique_items[val] = score
            
            sorted_rag_items = sorted(unique_items.items(), key=lambda item: item[1], reverse=True)
            result = "\n".join([f"- {val}" for val, score in sorted_rag_items])
            _log(f"SEARCH END (RAG): {len(sorted_rag_items)} items, total {(time.time()-t_start)*1000:.1f}ms")
            _log(f"RAG result preview: {result[:300]}")
            return result
        else:
            table_max_scores = {}
            for table, columns_data in collected_results.items():
                max_score_in_table = 0
                for column, values_with_scores in columns_data.items():
                    if values_with_scores:
                        max_score_in_column = values_with_scores[0][1]
                        if max_score_in_table < max_score_in_column:
                            max_score_in_table = max_score_in_column
                table_max_scores[table] = max_score_in_table

            sorted_tables = sorted(collected_results.items(), key=lambda item: table_max_scores.get(item[0], 0), reverse=True)

            output_lines = []
            for table, columns_data in sorted_tables:
                output_lines.append(f"【Table】：{table}")
                sorted_columns = sorted(columns_data.items(), key=lambda item: item[1][0][1] if item[1] else 0, reverse=True)
                for column, values_with_scores in sorted_columns:
                    output_lines.append(f"[{column}]: " + "、".join([v for v, s in values_with_scores]))
            
            result = "\n".join(output_lines)
            _log(f"SEARCH END (value): {len(sorted_tables)} tables, total {(time.time()-t_start)*1000:.1f}ms")
            _log(f"Value result preview: {result[:300]}")
            return result


# 全局检索器实例（懒加载）
_value_searcher = None
_kb_searcher = None
_searcher_lock = None

def _get_value_searcher():
    global _value_searcher, _searcher_lock
    import threading
    if _searcher_lock is None:
        _searcher_lock = threading.Lock()
    with _searcher_lock:
        if _value_searcher is None:
            _log("Loading value searcher for first time (mimic_iv indices)...")
            t0 = time.time()
            _value_searcher = DatabaseSearcher(
                tokenizer_path=config.TOKENIZER_PATH,
                indices_path=config.VALUE_INDICES_PATH,
                metadata_path=config.VALUE_METADATA_PATH,
            )
            _log(f"Value searcher fully loaded in {(time.time()-t0)*1000:.1f}ms")
        return _value_searcher

def _get_kb_searcher():
    global _kb_searcher, _searcher_lock
    import threading
    if _searcher_lock is None:
        _searcher_lock = threading.Lock()
    with _searcher_lock:
        if _kb_searcher is None:
            _log("Loading KB searcher for first time (knowledge base indices)...")
            t0 = time.time()
            _kb_searcher = DatabaseSearcher(
                tokenizer_path=config.TOKENIZER_PATH,
                indices_path=config.KB_INDICES_PATH,
                metadata_path=config.KB_METADATA_PATH,
            )
            _log(f"KB searcher fully loaded in {(time.time()-t0)*1000:.1f}ms")
        return _kb_searcher


def run_retrieval_router(question: str, ddl: str, selected_tables: List[str], history: List[Dict] = None) -> Dict[str, Optional[str]]:
    """
    检索路由：使用 DeepSeek Tool Calls 判断是否需要值检索和知识库检索
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    
    _log("=" * 80)
    _log("RETRIEVAL ROUTER START")
    _log(f"Question: {question}")
    _log(f"Selected tables: {selected_tables}")
    _log(f"DDL length: {len(ddl)} chars")
    _log(f"History turns: {len(history) if history else 0}")
    
    # 格式化历史对话
    history_context = ""
    if history and len(history) > 0:
        history_parts = []
        for i, turn in enumerate(history[-5:], 1):  # 最近5轮
            q = turn.get("question", "")
            sql = turn.get("sql", "")
            if q:
                history_parts.append(f"Q{i}: {q}")
                if sql:
                    history_parts.append(f"SQL{i}: {sql}")
        history_context = "\n".join(history_parts)
        _log(f"History context:\n{history_context}")
    else:
        _log("No conversation history")
    
    start_time = time.time()
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_values",
                "description": "Search for specific values in database columns. Use this when the user question contains entity names, diagnoses, medications, locations, or any specific string that might need fuzzy matching in the database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of value strings to search for. Each string should be a specific entity name, diagnosis, medication, or descriptive term. Use empty list [] if no value search is needed."
                        }
                    },
                    "required": ["queries"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "Search for business logic, formulas, definitions, or domain knowledge. Use this when the question involves calculations, derived metrics, business terms, or requires understanding of how to compute something from raw data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of knowledge keywords to search for. Use empty list [] if no knowledge search is needed."
                        }
                    },
                    "required": ["queries"]
                }
            }
        }
    ]
    
    prompt = f"""你是一个检索路由助手。根据用户问题和数据库Schema，判断是否需要进行值检索和知识库检索。

数据库Schema（已选表）：
{ddl}

已选表：{', '.join(selected_tables) if selected_tables else '全部表'}

对话历史：
{history_context if history_context else '无'}

当前问题：{question}

请分析：
1. **值检索（search_values）**：如果问题中包含具体的实体名称、诊断、药物、地点、检验项目、标本类型等需要在数据库中模糊匹配的具体字符串，调用此工具。
2. **知识库检索（search_knowledge）**：如果问题涉及以下任何情况，**必须调用**此工具：
   - 计算公式、业务规则（如"平均等待时间"、"住院天数"、"死亡率"）
   - 时间间隔计算（如"从X到Y的时间差"、"先后顺序"）
   - 专业术语定义、指标含义
   - 四舍五入、单位换算等数据处理规则
   - 任何需要理解"如何从原始数据计算某个指标"的问题
3. 如果都不需要，不调用任何工具。

注意：
- **两个工具可以同时调用**，互不影响
- 值检索找的是"数据库里存了什么具体值"
- 知识库检索找的是"这个业务指标应该怎么算"
- 当问题同时涉及具体值和计算逻辑时，**两个都调用**
"""
    
    _log(f"Router prompt length: {len(prompt)} chars")
    
    t_llm = time.time()
    _log(f"Calling LLM for routing decision (model={config.DEEPSEEK_MODEL}, base_url={config.DEEPSEEK_BASE_URL})")
    
    llm = ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0,
        max_tokens=512,
    )
    
    response = llm.bind_tools(tools).invoke([HumanMessage(content=prompt)])
    _log(f"LLM routing call took {(time.time()-t_llm)*1000:.1f}ms")
    _log(f"LLM response type: {type(response).__name__}")
    _log(f"LLM response content: {response.content}")
    
    # 解析 tool calls
    _log(f"Response has tool_calls attr: {hasattr(response, 'tool_calls')}")
    raw_tool_calls = getattr(response, 'tool_calls', None)
    _log(f"Raw tool_calls: {raw_tool_calls}")
    _log(f"Raw tool_calls type: {type(raw_tool_calls)}")
    
    # 兼容不同格式
    tool_calls = []
    if raw_tool_calls:
        for tc in raw_tool_calls:
            if isinstance(tc, dict):
                tool_calls.append(tc)
            elif hasattr(tc, 'name') and hasattr(tc, 'args'):
                tool_calls.append({"name": tc.name, "args": tc.args, "id": getattr(tc, 'id', '')})
            else:
                _log(f"Unknown tool call format: {type(tc)} = {tc}")
    
    _log(f"Tool calls detected: {len(tool_calls)}")
    
    for i, tc in enumerate(tool_calls):
        _log(f"  Tool call #{i+1}: name={tc.get('name')}, args={json.dumps(tc.get('args', {}), ensure_ascii=False)}")
    
    external_knowledge = None
    value_result = None
    
    for i, tc in enumerate(tool_calls):
        _log(f"Processing tool call #{i+1}...")
        func_name = tc.get("name", "")
        args = tc.get("args", {})
        queries = args.get("queries", [])
        _log(f"  func_name='{func_name}', queries={queries}, bool(queries)={bool(queries)}")
        
        if func_name == "search_values" and queries:
            _log(f"EXECUTING search_values with queries={queries}")
            t_search = time.time()
            try:
                _log("  Getting value searcher...")
                searcher = _get_value_searcher()
                _log("  Value searcher obtained, calling search()...")
                value_queries = list(queries)
                if question not in value_queries:
                    value_queries.insert(0, question)
                _log(f"  Value queries (with user question): {value_queries}")
                value_result = searcher.search(
                    queries=value_queries,
                    top_k=3,
                    threshold=0.2,
                    target_tables=selected_tables,
                    mode="value"
                )
                if value_result == "NULL":
                    value_result = None
                _log(f"search_values completed in {(time.time()-t_search)*1000:.1f}ms")
                _log(f"search_values result: {value_result}")
            except Exception as e:
                _log(f"search_values ERROR: {e}")
                import traceback
                _log(f"Traceback: {traceback.format_exc()}")
                value_result = None
        
        elif func_name == "search_knowledge" and queries:
            _log(f"EXECUTING search_knowledge with queries={queries}")
            t_search = time.time()
            try:
                _log("  Getting KB searcher...")
                kb_searcher = _get_kb_searcher()
                _log("  KB searcher obtained, calling search()...")
                kb_queries = list(queries)
                if question not in kb_queries:
                    kb_queries.insert(0, question)
                _log(f"  KB queries (with user question): {kb_queries}")
                external_knowledge = kb_searcher.search(
                    queries=kb_queries,
                    top_k=3,
                    threshold=0.2,
                    mode="RAG",
                    rag_columns=["content"]
                )
                if external_knowledge == "NULL":
                    external_knowledge = None
                _log(f"search_knowledge completed in {(time.time()-t_search)*1000:.1f}ms")
                _log(f"search_knowledge result: {external_knowledge}")
            except Exception as e:
                _log(f"search_knowledge ERROR: {e}")
                import traceback
                _log(f"Traceback: {traceback.format_exc()}")
                external_knowledge = None
        else:
            _log(f"  SKIPPING tool call #{i+1}: func_name='{func_name}', queries={queries}")
    
    if not tool_calls:
        _log("No tool calls made - LLM decided no retrieval needed")
    
    elapsed = time.time() - start_time
    _log(f"RETRIEVAL ROUTER TOTAL: {elapsed:.2f}s")
    _log(f"Final external_knowledge: {external_knowledge}")
    _log(f"Final Value: {value_result}")
    _log("=" * 80)
    
    return {
        "external_knowledge": external_knowledge,
        "Value": value_result
    }
