"""
表选择模块：根据用户问题和对话历史，智能选择需要数据库表
解决大数据库DDL过长超过上下文窗口的问题
"""
import re
import json
import time
import sys
import threading
from typing import Dict, List, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

import config
from prompts import get_tablePrompt_d, build_ddl_string
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'assets', 'mytoken'))
from deepseek_tokenizer import get_token_count

LOG_FILE = "/amax/storage/nfs/qcjySONG/SQL_Dog/frontend.log"
_log_lock = threading.Lock()
_call_counter = 0
_call_lock = threading.Lock()

def _ts():
    return time.strftime("%H:%M:%S.", time.localtime()) + f"{time.time() % 1:.3f}"[2:]

def log_debug(msg):
    line = f"[{_ts()}] {msg}"
    with _log_lock:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
            f.flush()
    print(line, flush=True)

# Pre-load tokenizer once at module level to avoid repeated disk I/O
try:
    _tokenizer_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'mytoken')
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(_tokenizer_dir, trust_remote_code=False)
    def _fast_token_count(text: str) -> int:
        return len(_tokenizer.encode(text))
    log_debug(f"[TABLE_SELECTOR] Tokenizer pre-loaded successfully")
except Exception as e:
    log_debug(f"[TABLE_SELECTOR] Tokenizer pre-load failed: {e}, falling back to deepseek_tokenizer")
    _tokenizer = None
    def _fast_token_count(text: str) -> int:
        return get_token_count(text)

LOG_FILE = "/amax/storage/nfs/qcjySONG/SQL_Dog/frontend.log"
_log_lock = threading.Lock()
_call_counter = 0
_call_lock = threading.Lock()

def _ts():
    return time.strftime("%H:%M:%S.", time.localtime()) + f"{time.time() % 1:.3f}"[2:]

def log_debug(msg):
    line = f"[{_ts()}] {msg}"
    with _log_lock:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
            f.flush()
    print(line, flush=True)


def extract_tables_from_response(response: str) -> List[str]:
    t0 = time.time()
    tables = []
    
    # 尝试匹配 ```json ... ``` 格式
    json_match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1).strip())
            if 'table' in data and isinstance(data['table'], list):
                tables.extend(data['table'])
        except json.JSONDecodeError:
            pass
    
    # 如果没找到，尝试直接解析整个响应
    if not tables:
        try:
            data = json.loads(response.strip())
            if 'table' in data and isinstance(data['table'], list):
                tables.extend(data['table'])
        except json.JSONDecodeError:
            pass
    
    # 保底方案：提取所有看起来像表名的字符串
    if not tables:
        table_matches = re.findall(r'["\']([a-zA-Z0-9_]+)["\']', response)
        tables.extend(table_matches)
    
    elapsed = time.time() - t0
    log_debug(f"  [extract_tables] regex+json parse took {elapsed*1000:.1f}ms -> {tables}")
    return list({t.lower() for t in tables})


def call_table_selection_llm(db_details: str, question: str, history_str: str) -> Set[str]:
    global _call_counter
    with _call_lock:
        _call_counter += 1
        call_id = _call_counter
    
    log_debug(f"  [call #{call_id}] START (thread={threading.current_thread().name})")
    t0 = time.time()
    
    t1 = time.time()
    prompt = get_tablePrompt_d(db_details=db_details, question=question, history=history_str)
    log_debug(f"  [call #{call_id}] get_tablePrompt_d took {(time.time()-t1)*1000:.1f}ms (prompt len={len(prompt)})")
    
    t2 = time.time()
    llm = ChatOpenAI(
        model=config.TABLE_LLM_MODEL,
        api_key=config.TABLE_LLM_API_KEY,
        base_url=config.TABLE_LLM_BASE_URL,
        temperature=0.1,
        max_tokens=512,
        n=config.TABLE_SELECTION_N,
    )
    log_debug(f"  [call #{call_id}] ChatOpenAI() init took {(time.time()-t2)*1000:.1f}ms (model={config.TABLE_LLM_MODEL}, n={config.TABLE_SELECTION_N})")
    
    t3 = time.time()
    response = llm.invoke([HumanMessage(content=prompt)])
    llm_invoke_ms = (time.time() - t3) * 1000
    log_debug(f"  [call #{call_id}] llm.invoke() took {llm_invoke_ms:.1f}ms")
    
    log_debug(f"  [call #{call_id}] Response type: {type(response).__name__}")
    
    all_tables = set()
    
    t4 = time.time()
    if hasattr(response, 'content'):
        raw_text = response.content
        log_debug(f"  [call #{call_id}] RAW_OUTPUT === BEGIN ===")
        for line in raw_text.split('\n'):
            log_debug(f"  [call #{call_id}] RAW | {line}")
        log_debug(f"  [call #{call_id}] RAW_OUTPUT === END ===")
        tables = extract_tables_from_response(raw_text)
        all_tables.update(tables)
    elif hasattr(response, 'generations'):
        for i, choice in enumerate(response.generations):
            for j, gen in enumerate(choice):
                raw_text = gen.text
                log_debug(f"  [call #{call_id}] RAW_OUTPUT === BEGIN choice={i} gen={j} ===")
                for line in raw_text.split('\n'):
                    log_debug(f"  [call #{call_id}] RAW | {line}")
                log_debug(f"  [call #{call_id}] RAW_OUTPUT === END ===")
                tables = extract_tables_from_response(raw_text)
                all_tables.update(tables)
    else:
        raw_text = str(response)
        log_debug(f"  [call #{call_id}] RAW_OUTPUT === BEGIN ===")
        for line in raw_text.split('\n'):
            log_debug(f"  [call #{call_id}] RAW | {line}")
        log_debug(f"  [call #{call_id}] RAW_OUTPUT === END ===")
        tables = extract_tables_from_response(raw_text)
        all_tables.update(tables)
    
    log_debug(f"  [call #{call_id}] extract+parse took {(time.time()-t4)*1000:.1f}ms")
    log_debug(f"  [call #{call_id}] Final tables: {all_tables}")
    log_debug(f"  [call #{call_id}] TOTAL took {(time.time()-t0)*1000:.1f}ms")
    return all_tables


def split_ddl_chunks(ddl_dict: Dict[str, str], max_tokens: int = 2048) -> List[Dict[str, str]]:
    t0 = time.time()
    all_tables = list(ddl_dict.items())
    chunks = []
    current_chunk = {}
    current_tokens = 0
    
    for table_name, ddl_sql in all_tables:
        table_tokens = _fast_token_count(ddl_sql) + 2
        if current_tokens + table_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = {}
            current_tokens = 0
        current_chunk[table_name] = ddl_sql
        current_tokens += table_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
    
    log_debug(f"  [split_ddl_chunks] took {(time.time()-t0)*1000:.1f}ms -> {len(chunks)} chunks")
    return chunks


def select_tables(ddl_dict: Dict[str, str], question: str, history: List[Dict]) -> List[str]:
    t_select_start = time.time()
    log_debug(f"[select_tables] START")
    
    t0 = time.time()
    history_str = ""
    for i, turn in enumerate(history, 1):
        history_str += f"<Question {i}>: {turn.get('question', '')}\n"
        if turn.get('sql'):
            history_str += f"<SQL {i}>: {turn['sql']}\n\n"
    log_debug(f"[select_tables] format_history took {(time.time()-t0)*1000:.1f}ms")
    
    t1 = time.time()
    full_ddl = build_ddl_string(ddl_dict)
    log_debug(f"[select_tables] build_ddl_string took {(time.time()-t1)*1000:.1f}ms (len={len(full_ddl)})")
    
    t2 = time.time()
    total_tokens = _fast_token_count(full_ddl)
    log_debug(f"[select_tables] get_token_count took {(time.time()-t2)*1000:.1f}ms (tokens={total_tokens})")
    
    if total_tokens <= 2048:
        log_debug(f"[select_tables] DDL token: {total_tokens} <= 2048, direct call")
        selected_tables = call_table_selection_llm(full_ddl, question, history_str)
        log_debug(f"[select_tables] LLM returned: {selected_tables}")
        log_debug(f"[select_tables] TOTAL took {(time.time()-t_select_start)*1000:.1f}ms")
        return list(selected_tables)
    
    log_debug(f"[select_tables] DDL token: {total_tokens} > 2048, chunking")
    t3 = time.time()
    ddl_chunks = split_ddl_chunks(ddl_dict, max_tokens=2048)
    log_debug(f"[select_tables] split_ddl_chunks took {(time.time()-t3)*1000:.1f}ms")
    
    all_selected_tables = set()
    
    t4 = time.time()
    with ThreadPoolExecutor(max_workers=len(ddl_chunks)) as executor:
        futures = []
        for i, chunk in enumerate(ddl_chunks):
            chunk_ddl = build_ddl_string(chunk)
            log_debug(f"[select_tables] chunk {i}: {len(chunk)} tables, ddl len={len(chunk_ddl)}")
            future = executor.submit(call_table_selection_llm, chunk_ddl, question, history_str)
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                chunk_tables = future.result()
                all_selected_tables.update(chunk_tables)
            except Exception as e:
                log_debug(f"[select_tables] ERROR chunk call failed: {e}")
    
    log_debug(f"[select_tables] ThreadPoolExecutor total took {(time.time()-t4)*1000:.1f}ms")
    
    t5 = time.time()
    valid_tables = [t for t in all_selected_tables if t in ddl_dict]
    log_debug(f"[select_tables] validate_tables took {(time.time()-t5)*1000:.1f}ms")
    
    if not valid_tables:
        log_debug(f"[select_tables] WARN: no valid tables, fallback to all")
        log_debug(f"[select_tables] TOTAL took {(time.time()-t_select_start)*1000:.1f}ms")
        return list(ddl_dict.keys())
    
    log_debug(f"[select_tables] Final valid tables: {valid_tables}")
    log_debug(f"[select_tables] TOTAL took {(time.time()-t_select_start)*1000:.1f}ms")
    return valid_tables


def filter_ddl_by_tables(ddl_dict: Dict[str, str], selected_tables: List[str]) -> str:
    filtered_ddl = {}
    for table in selected_tables:
        if table in ddl_dict:
            filtered_ddl[table] = ddl_dict[table]
    return build_ddl_string(filtered_ddl)


if __name__ == "__main__":
    test_ddl = {
        "patients": "CREATE TABLE patients (subject_id INTEGER, gender TEXT);",
        "admissions": "CREATE TABLE admissions (subject_id INTEGER, hadm_id INTEGER);",
        "prescriptions": "CREATE TABLE prescriptions (subject_id INTEGER, drug TEXT);",
        "labevents": "CREATE TABLE labevents (subject_id INTEGER, itemid INTEGER, value FLOAT);",
    }
    selected = select_tables(test_ddl, "查询所有男性患者的处方记录", [])
    print(f"测试选中的表: {selected}")
