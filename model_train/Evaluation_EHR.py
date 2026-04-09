import sqlite3
import json
import os
import re
import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures # 引入并发模块
from concurrent.futures import as_completed # 核心修复库
import threading # 引入锁
import json
from collections import defaultdict
import os

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================
# 0. 配置区域 (Configuration)
# ==============================

# # --- VLLM 配置 ---
VLLM_BASE_URL = "http://localhost:8192/v1"
VLLM_API_KEY = "not-used"
LLM_TIMEOUT_SECONDS = 200 
# 注意：OpenAI Client 在多线程中通常是线程安全的，但在极高并发下建议每个线程新建，
# 这里的 8 并发通常可以直接共享全局 CLIENT。
CLIENT = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)




# 相对路径配置
MODEL_ID = os.path.join(SCRIPT_DIR, "MODEL", "SQL_Dog_DPO")
INPUT_FILE = os.path.join(SCRIPT_DIR, "DATA", "EHR24_TEST_SUCCESS.json")
DB_FILE = os.path.join(SCRIPT_DIR, "EHR_DB", "mimic_iv", "mimic_iv.sqlite")
TABLES_JSON_PATH = os.path.join(SCRIPT_DIR, "EHR_DB", "mimic_iv", "tables.json")
Table_results = os.path.join(SCRIPT_DIR, "EHR24_test_table.jsonl")
Value_results = os.path.join(SCRIPT_DIR, "EHR24_test_Value_Retrieval.jsonl")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "Output", "SQL_Dog_DPO_n=1.jsonl")



# ---------- 核心开关 ----------
ENABLE_TABLE_SELECTION = False #False时则为选取全量表
ENABLE_VALUE_RETRIEVAL = True
ENABLE_SELF_CONSISTENCY = False #是否开启一致性
SC_SAMPLES = 8
SC_TEMPERATURE = 0.8 
NORMAL_TEMPERATURE = 0

# ---------- 并发配置 (新增) ----------
MAX_WORKERS =1   # 同时运行的任务数量
max_token=2048

# ==============================
# 1. 依赖导入与工具初始化
# ==============================
try:
    from all_prompt import *
    if ENABLE_SELF_CONSISTENCY:
        from self_consistency import SelfConsistencySelectorV2
except ImportError as e:
    print(f"⚠️ 警告: 缺少部分模块，将使用 Mock 函数。错误: {e}")
    def get_initPrompt(schema, q, Value=None): 
        val_str = f"\nValues: {Value}" if Value else ""
        return f"{schema}{val_str}\n\nQuestion: {q}\nGenerate SQL:"
    def get_followupPrompt(schema, q, h, Value=None): 
        val_str = f"\nValues: {Value}" if Value else ""
        return f"{schema}{val_str}\n\nHistory:\n{h}\n\nQuestion: {q}\nGenerate SQL:"
    def return_keyword(text): return [text]
    class DatabaseSearcher:
        def __init__(self, **kwargs): pass
        def search(self, queries, target_tables): return "Mock Value Info"
    class SelfConsistencySelectorV2:
        def __init__(self, db_path, tables_json_path): pass
        def select_best_query(self, sqls): return sqls[0] if sqls else "-- Error"


selector = None
if ENABLE_SELF_CONSISTENCY:
    print("Initializing SelfConsistencySelectorV2...")
    selector = SelfConsistencySelectorV2(db_path=DB_FILE, tables_json_path=TABLES_JSON_PATH)

# ==============================
# 2. 核心辅助函数
# ==============================
    
import sqlite3

def execute_sqlite_query_test(db_path: str, sql: str) -> None:
    """
    执行SQLite查询语句，有错误则打印错误
    
    Args:
        db_path: SQLite数据库文件路径
        sql: 要执行的SQL查询语句
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        # 获取并打印结果（可选）
        results = cursor.fetchall()
        #print(f"查询成功，返回 {len(results)} 行结果")
        
    except sqlite3.Error as e:
        print(f"SQL执行错误: {e}")
        
    except Exception as e:
        print(f"其他错误: {e}")
        
    finally:
        if conn:
            conn.close()

def load_results_as_dict(file_path):
    data_map = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                iid = item.get("interaction_id")
                if iid:
                    data_map[iid].append(item)
            except: pass
    
    # 预先给所有列表排序
    for iid in data_map:
        data_map[iid].sort(key=lambda x: x.get("turn_idx", 0))
    return data_map

from typing import List, Optional

def get_pred_tables_by_question(history: List[dict], question_text: str) -> Optional[List[str]]:
    """
    在 history 列表中查找指定的 question，并返回对应的 pred_tables。
    
    Args:
        history: 包含对话记录的列表 
        question_text: 要查找的问题文本
        
    Returns:
        匹配到的 pred_tables 列表，如果未找到则返回 None
    """
    # 去除输入问题的首尾空格，防止因复制粘贴导致的格式差异
    target_q = question_text.strip()
    
    for item in history:
        # 获取当前项的问题 (同时也去除空格)
        current_q = item.get("question", "").strip()
        
        # 字符串匹配
        if current_q == target_q:
            # 找到匹配项，返回 pred_tables，如果该字段不存在则返回空列表
            return item.get("pred_tables", [])
            
    return []

def get_pred_val_by_question(history: List[dict], question_text: str) -> Optional[str]:
    """
    在 history 列表中查找指定的 question，并返回对应的 val 字符串。

    Args:
        history: 包含对话记录的列表
        question_text: 要查找的问题文本

    Returns:
        匹配到的 val 字符串；若未找到或字段不存在，返回 None
    """
    # 去除输入问题的首尾空格，防止因复制粘贴导致的格式差异
    target_q = question_text.strip()

    for item in history:
        # 获取当前项的问题（同样去除空格）
        current_q = item.get("question", "").strip()

        # 字符串精确匹配
        if current_q == target_q:
            return item.get("val_info_str"),item.get("KG_info_str")  # 不存在则自然返回 None

    return None


def generate_ddl_dict(db_path: str, schema_to_nl_mapping: Dict) -> Dict[str, str]:
    """生成包含外键的完整 DDL 字典"""
    ddl_map = {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    nl_map_lower = {k.lower(): v for k, v in schema_to_nl_mapping.items()}
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    all_tables = [r["name"] for r in cursor.fetchall()]

    for table in all_tables:
        cursor.execute(f"PRAGMA table_info('{table}')")
        columns = cursor.fetchall()
        cursor.execute(f"SELECT * FROM '{table}' LIMIT 3")
        rows = cursor.fetchall()
        examples_by_col = {col["name"]: [] for col in columns}
        for row in rows:
            for col in columns:
                val = row[col["name"]]
                if len(examples_by_col[col["name"]]) < 3 and val is not None:
                    val_str = str(val)[:50]
                    if val_str not in examples_by_col[col["name"]]:
                        examples_by_col[col["name"]].append(val_str)

        lines = [f"CREATE TABLE {table} ("]
        for col in columns:
            name = col["name"]
            dtype = col["type"]
            nl_desc = nl_map_lower.get(f"{table}.{name}".lower(), "")
            ex_str = json.dumps(examples_by_col[name], ensure_ascii=False)
            comment = f"-- {nl_desc} ,example: {ex_str}" if nl_desc else f"-- example: {ex_str}"
            lines.append(f"    {name} {dtype}, {comment}")
            
        cursor.execute(f"PRAGMA foreign_key_list('{table}')")
        fks = cursor.fetchall()
        if fks:
            for fk in fks:
                lines.append(f"    FOREIGN KEY ({fk[3]}) REFERENCES {fk[2]}({fk[4]})")
        lines.append(");")
        ddl_map[table.lower()] = "\n".join(lines)
    conn.close()
    return ddl_map

def extract_last_sql_block(text: str) -> Optional[str]:
    """提取最后一个 ```sql ... ``` 代码块，如果未找到则返回 None"""
    if not text:
        return None

    # 步骤 1: 预处理文本
    # 使用 re.sub 将从开头到最后一个 '</think>' 的所有内容替换为空字符串
    # re.DOTALL 标志确保 '.*' 可以匹配包括换行符在内的任意字符
    processed_text = re.sub(r'^.*</think>', '', text, flags=re.DOTALL)

    # 步骤 2: 在处理后的文本上提取 SQL 代码块
    matches = re.findall(r"```sql\s*(.*?)```", processed_text, re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    return None

# ==============================
# 3. 单条目处理函数 (Thread Worker)
# ==============================

def process_single_interaction(item: Dict, item_idx: int, ddl_dict: Dict, all_table_names: List[str],all_Table_results,all_val_LSH_results):
    """
    处理单个交互 item 的完整逻辑。
    包含针对贪婪解码模式的自动重试机制：
    如果不启用自一致性，且无法从输出中提取 SQL (extract_last_sql_block 返回 None)，则重试最多 10 次。
    """
    interaction_id = item.get('id', f"idx_{item_idx}")
    Table_name_pre=all_Table_results.get(interaction_id, [])
    question_LSH=all_val_LSH_results.get(interaction_id, [])
    questions = item.get('question', [])
    gold_sqls = item.get('seqsql', [])
    
    # --- 修改点 1: 外部知识 Header 的生成方式需与参考代码完全一致 ---
    external_knowledge_list = item.get('external_knowledge', [])
    external_knowledge = "\n".join(external_knowledge_list)
    ek_header = f"{external_knowledge}\n" if external_knowledge else ""
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    output_item = {
        "id": interaction_id,
        "Gold_SQL": gold_sqls
    }
    
    history_str = "" 
    history_questions_list = [] 
    
    for turn_idx, question in enumerate(questions):
        history_questions_list.append(question)
        search_query_str = " | ".join(history_questions_list)
        
        # B. Table Selection
        target_tables = found_tables if (ENABLE_TABLE_SELECTION and (found_tables := get_pred_tables_by_question(history=Table_name_pre, question_text=question))) else all_table_names
        
        # --- 修改点 2: Schema 中不再直接拼接 ek_header，保持纯净 ---
        current_schema_str = "\n\n".join([ddl_dict[t] for t in target_tables if t in ddl_dict])
        
        # C. Value Retrieval
        val_info_str = None
        KG_info_str=None
        if ENABLE_VALUE_RETRIEVAL:
            try:
                val_info_str,KG_info_str=get_pred_val_by_question(history=question_LSH,question_text=question)
                if val_info_str =="未找到相关结果。":
                    val_info_str=None
            except Exception as e:
                val_info_str = None

        # --- 修改点 3: 构造 combined_ek，其中 KG_info_str 按要求置为空字符串 ---
        if KG_info_str=="NULL":
            KG_info_str=None

        #KG_info_str=None
        #val_info_str=None #强制修改

        combined_ek = (ek_header or "") + (KG_info_str or "")
        combined_ek = combined_ek.strip() if combined_ek.strip() else None

        # D. Prompt Construction
        # --- 修改点 4: 将 external_knowledge 作为参数传入 prompt 函数 ---
        if turn_idx == 0:
            if "OmniSQL" in MODEL_ID:
                prompt=OmniSQL_Prompt(current_schema_str,question)
            else:
                prompt = get_initPrompt_EHR24(current_schema_str, question, Value=val_info_str)
        else:
            prompt = get_followupPrompt(current_schema_str, question, history_str, Value=val_info_str, external_knowledge=combined_ek)
        
        # ==========================================
        # E. LLM Inference (带基于提取检测的重试机制)
        # ==========================================
        llm_raw_outputs = []
        extracted_sqls = []
        best_sql = ""
        turn_input_tokens = 0
        turn_output_tokens = 0
        
        # 核心逻辑：如果开启自一致性，交给 SC 采样(loop 1次)；如果不开启，贪婪模式下重试 10 次直到提取出 SQL
        max_retries = 10 if not ENABLE_SELF_CONSISTENCY else 1
        
        for attempt in range(max_retries):
            try:
                num_samples = SC_SAMPLES if ENABLE_SELF_CONSISTENCY else 1
                temp = SC_TEMPERATURE if ENABLE_SELF_CONSISTENCY else NORMAL_TEMPERATURE
                
                SYS_PROMPT="""
Please follow the following format for your response:
<thinking>
- **Analyze the User's Intent:**...
- **Consider Conversation History:**...
- **Map to Schema:**...
- **Formulate a Plan:**...
</thinking>
**Final SQL:**
```sql
-- Your SQL query
```
"""
                response = CLIENT.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                       # {"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    n=num_samples,
                    temperature=temp,
                    max_tokens=max_token
                )
                
                # 累计 Token (包括失败重试的消耗)
                if hasattr(response, 'usage') and response.usage:
                    in_tokens = response.usage.prompt_tokens
                    out_tokens = response.usage.completion_tokens
                    turn_input_tokens += in_tokens
                    turn_output_tokens += out_tokens
                    total_input_tokens += in_tokens
                    total_output_tokens += out_tokens
                
                # 清空本轮临时的列表，准备接收新数据
                llm_raw_outputs = []
                extracted_sqls = []
                
                current_batch_has_sql = False # 标记本批次是否成功提取到 SQL
                
                for choice in response.choices:
                    content = choice.message.content
                    print(content)
                    llm_raw_outputs.append(content)
                    
                    # 关键调用：extract_last_sql_block 现在返回 None 表示失败
                    sql_ex = extract_last_sql_block(content)
                    extracted_sqls.append(sql_ex)
                    
                    if sql_ex is not None:
                        current_batch_has_sql = True
                
                # --- 重试判定逻辑 ---
                if not ENABLE_SELF_CONSISTENCY:
                    # 贪婪模式：如果本轮生成的文本无法提取出 SQL (sql_ex is None)
                    if not current_batch_has_sql:
                        if attempt < max_retries - 1:
                            # 还有重试机会，继续下一轮循环
                            # print(f"⚠️ [ID: {interaction_id}] Attempt {attempt+1} failed to extract SQL. Retrying...")
                            continue 
                        else:
                            # 此时已是最后一次尝试且依然为 None
                            best_sql = "SELECT 'Error: No SQL block found after retries';"
                    else:
                        # 成功提取到 SQL，跳出循环
                        break
                else:
                    # SC 模式：不进行重试，直接跳出，后续逻辑处理
                    break
                    
            except Exception as e:
                # 捕获 API 错误或网络波动
                if attempt < max_retries - 1:
                    print(f"⚠️ Error ID {interaction_id} (Attempt {attempt+1}) Exception: {e}. Retrying...")
                    continue
                else:
                    # 最后一次尝试也失败
                    best_sql = f"SELECT 'Error: {str(e)}';"
                    llm_raw_outputs.append(str(e))
                    break

        # ==========================================
        # F. Selector & Post-processing
        # ==========================================
        if ENABLE_SELF_CONSISTENCY and selector:
            # 过滤掉 None 的结果
            valid_sqls = [s for s in extracted_sqls if s is not None and "select" in s.lower()]
            
            # 兜底：如果全部提取失败
            if not valid_sqls: 
                valid_sqls = [s for s in extracted_sqls if s is not None]
                if not valid_sqls: 
                    valid_sqls = llm_raw_outputs # 实在没办法，用原始文本
            
            best_sql = selector.select_best_query(valid_sqls)
        else:
            # 贪婪模式
            if not best_sql: # 如果异常处理块没有赋值
                first_sql = extracted_sqls[0] if extracted_sqls else None
                if first_sql is not None:
                    best_sql = first_sql
                    execute_sqlite_query_test(db_path=DB_FILE,sql=best_sql)
                else:
                    # 这里对应重试了10次依然全部返回 None 的情况
                    best_sql = "SELECT 'Error: Model failed to generate SQL format';"
        
        turn_key = f"turn_{turn_idx+1}"
        output_item[turn_key] = {
            "question": question,
            "Prompt": prompt,
            "LLM_out": llm_raw_outputs,
            "Pre_SQL": extracted_sqls,
            "best_sql": best_sql,
            "token_usage": {"input": turn_input_tokens, "output": turn_output_tokens}
        }
        
        new_block = f"<Question {turn_idx+1}>: {question}\n<SQL {turn_idx+1}>: {best_sql}\n"
        history_str = (history_str + f"\n---\n{new_block}") if history_str else new_block
    
    output_item["total_token_stats"] = {
        "total_input": total_input_tokens,
        "total_output": total_output_tokens,
        "grand_total": total_input_tokens + total_output_tokens
    }
    
    return output_item


# ==============================
# 4. 主程序入口 (Multi-threaded)
# ==============================

def run_inference():
    NL_MAPPING={}
    # 4.1 准备 Schema (只读资源)
    print("Generating DDL Dictionary...")
    # 注意：generate_ddl_dict 函数需在外部定义好，此处直接调用
    ddl_dict = generate_ddl_dict(DB_FILE, NL_MAPPING)
    all_table_names = list(ddl_dict.keys())
    
    # 4.2 加载原始数据
    with open(INPUT_FILE, 'r') as f:
        full_data = json.load(f)
    
    total_items = len(full_data)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # ---------------------------------------------------------
    # [NEW] 断点检测逻辑
    # ---------------------------------------------------------
    processed_count = 0
    # processed_ids = set() # 如果需要更严格的去重可以启用
    
    if os.path.exists(OUTPUT_FILE):
        print(f"检测到输出文件 {OUTPUT_FILE}，正在扫描已完成的任务...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    # 尝试解析每一行，确保是完整的 JSON
                    record = json.loads(line)
                    processed_count += 1
                except json.JSONDecodeError:
                    print("⚠️ 发现损坏的行，将从此处继续。")
                    break
        print(f"✅ 已完成任务数: {processed_count}/{total_items}")
    else:
        print("未检测到旧的输出文件，将开始新的任务。")

    if processed_count >= total_items:
        print("所有任务已完成！无需执行。")
        return

    # 4.3 确定剩余任务
    remaining_data = full_data[processed_count:] 
    start_index = processed_count # 保持原始索引 ID 一致
    
    print("Loading auxiliary results...")
    # 加载辅助文件 (Table results, Value results)
    all_Table_results = load_results_as_dict(Table_results)
    all_val_LSH_results = load_results_as_dict(Value_results)
    
    print(f"🚀 将启动 {MAX_WORKERS} 个线程处理剩余的 {len(remaining_data)} 个任务。")

    # =========================================================
    # 全局预配置数据库 (优化 SQLite 写入性能，只做一次)
    # =========================================================
    print("正在预配置数据库模式...")
    try:
        init_conn = sqlite3.connect(DB_FILE, timeout=10)
        init_conn.execute("PRAGMA journal_mode = OFF;") 
        init_conn.execute("PRAGMA synchronous = OFF;")
        init_conn.close()
        print("数据库预配置完成。")
    except Exception as e:
        print(f"⚠️ 数据库预配置警告 (非致命): {e}")

    # =========================================================
    # 4.4 多线程执行 & 乱序实时写入
    # =========================================================
    
    # 建立写入锁
    write_lock = threading.Lock()
    
    # 确定写入模式：如果有历史数据则追加 'a'，否则新建 'w'
    write_mode = 'a' if processed_count > 0 else 'w'
    
    with open(OUTPUT_FILE, write_mode, encoding='utf-8') as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            
            # 1. 提交所有任务
            # 使用字典映射 future -> task_id，以便报错时知道是谁挂了
            future_to_id = {}
            
            print("Submitting tasks to executor...")
            for idx, item in enumerate(remaining_data):
                real_idx = start_index + idx 
                
                future = executor.submit(
                    process_single_interaction, 
                    item, 
                    real_idx, 
                    ddl_dict, 
                    all_table_names, 
                    all_Table_results, 
                    all_val_LSH_results
                )
                
                # 记录 ID 用于进度条显示或报错
                item_id = item.get('id', f"idx_{real_idx}")
                future_to_id[future] = item_id

            # 2. 使用 as_completed 处理完成的任务
            # 这里是关键：谁先跑完，谁先 yield 出来，不会被慢任务阻塞
            pbar = tqdm(total=len(remaining_data), desc="Processing (Real-time Write)")
            
            for future in as_completed(future_to_id):
                task_id = future_to_id[future]
                try:
                    result = future.result()
                    
                    # 加锁写入，保证线程安全
                    with write_lock:
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush() # 【关键】强制刷新缓冲区，确保磁盘文件立即更新
                    
                except Exception as exc:
                    # 捕获线程内部未处理的异常，防止程序静默退出
                    tqdm.write(f"\n❌ Task {task_id} generated an exception: {exc}")
                    import traceback
                    tqdm.write(traceback.format_exc())
                finally:
                    pbar.update(1)
            
            pbar.close()

    print("✅ Inference Completed.")

if __name__ == '__main__':
    run_inference()