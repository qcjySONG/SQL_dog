import json
import sqlite3
import threading
import time
import os
import itertools
from enum import Enum
from typing import List, Optional, Any
from collections import defaultdict, Counter

# ============================================================
# 0. 核心执行一致性评估算法 (列顺序无关、行顺序无关)
# ============================================================
def robust_ex_eval(sql1: str, sql2: str, db_path: str) -> bool:
    """严谨版：支持严格的全局列顺序无关、行顺序无关"""
    def fetch_results(sql):
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10) as conn:
            conn.text_factory = lambda b: b.decode(errors="ignore")
            return conn.execute(sql).fetchall()
    try:
        res1 = fetch_results(sql1)
        res2 = fetch_results(sql2)
    except Exception:
        return False

    if not res1 and not res2: return True
    if not res1 or not res2: return False
    if len(res1) != len(res2): return False

    num_cols = len(res1[0])
    if len(res2[0]) != num_cols: return False

    # 快速拒绝：打平所有元素进行统计比对
    flat_counter1 = Counter(x for row in res1 for x in row)
    flat_counter2 = Counter(x for row in res2 for x in row)
    if flat_counter1 != flat_counter2:
        return False

    if num_cols == 1:
        return Counter(res1) == Counter(res2)

    # 全局列组合比对
    c1 = Counter(res1)
    for perm in itertools.permutations(range(num_cols)):
        res2_permuted = [tuple(row[i] for i in perm) for row in res2]
        if c1 == Counter(res2_permuted):
            return True

    return False

def evaluate_single_sql(sql1: str, sql2: str, db_path: str, tables_json_path: str = None) -> dict:
    """提供给外部或测试代码调用的评估函数"""
    is_exec = robust_ex_eval(sql1, sql2, db_path)
    # 注意：这里返回测试代码期望的 "exec" 键
    return {"exec": 1 if is_exec else 0}


# ============================================================
# 1. SQL 执行（带超时）
# ============================================================
class SQLExecutionResultType(Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"

class SQLExecutionResult:
    def __init__(self, db_path, query, result_type, result_cols, result, error_message):
        self.db_path = db_path
        self.query = query
        self.result_type = result_type
        self.result_cols = result_cols
        self.result = result
        self.error_message = error_message

class ExecuteSQLThread(threading.Thread):
    def __init__(self, db_path: str, query: str, timeout: int):
        super().__init__()
        self.db_path = db_path
        self.query = query
        self.timeout = timeout
        self.result = None
        self.result_cols = None
        self.exception = None
        self.stop_event = threading.Event()
        self.conn = None

    def run(self):
        try:
            self.conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", 
                uri=True, 
                timeout=self.timeout, 
                check_same_thread=False
            )
            self.conn.execute("PRAGMA mmap_size = 2147483648;") 
            self.conn.execute("PRAGMA query_only = 1;")
            self.conn.set_progress_handler(lambda: 1 if self.stop_event.is_set() else 0, 100)
            
            cursor = self.conn.cursor()
            cursor.execute(self.query)
            self.result = cursor.fetchall()
            self.result_cols = [desc[0] for desc in cursor.description] if cursor.description else[]
            
        except Exception as e:
            self.exception = e
        finally:
            if self.conn:
                self.conn.close()

def execute_sql_with_timeout(db_path: str, query: str, timeout: int = 30) -> SQLExecutionResult:
    thread = ExecuteSQLThread(db_path, query, timeout)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        thread.stop_event.set()
        thread.join(1)
        return SQLExecutionResult(db_path, query, SQLExecutionResultType.TIMEOUT, None, None, "Timeout")

    if thread.exception:
        return SQLExecutionResult(db_path, query, SQLExecutionResultType.ERROR, None, None, str(thread.exception))

    return SQLExecutionResult(db_path, query, SQLExecutionResultType.SUCCESS, thread.result_cols, thread.result, None)


# ============================================================
# 2. 自一致性选择器 (纯 EX + Time Penalty)
# ============================================================
class SelfConsistencySelectorV2:
    EXECUTION_TIME_REPEAT = 3
    SQL_EXECUTION_TIMEOUT = 30

    def __init__(self, db_path: str, tables_json_path: str = None,
                 alpha: float = 1.0, gamma: float = 0.1, **kwargs):
        self.db_path = db_path
        self.tables_json_path = tables_json_path
        self.alpha = alpha
        self.gamma = gamma

    def _execute_sql(self, sql_query: str) -> SQLExecutionResult:
        return execute_sql_with_timeout(self.db_path, sql_query, timeout=self.SQL_EXECUTION_TIMEOUT)

    def _is_valid_execution_result(self, result: Optional[List[Any]]) -> bool:
        if result is None or not result:
            return False
        if len(result) == 1 and (result[0] == (None,) or not result[0]):
            return False
        return True

    def _measure_execution_time(self, sql_query: str) -> float:
        total_time = 0
        success = 0
        for _ in range(self.EXECUTION_TIME_REPEAT):
            start = time.perf_counter()
            res = self._execute_sql(sql_query)
            end = time.perf_counter()
            if res.result_type == SQLExecutionResultType.SUCCESS:
                total_time += (end - start)
                success += 1
            else:
                return float("inf")
        return total_time / success if success > 0 else float("inf")

    def _get_canonical_result_key(self, result: List[tuple]) -> tuple:
        """
        生成结果的标准 Key，保证：
        1. 行顺序无关 (对结果进行排序)
        2. 保留重复行 (替代原版易错的 frozenset)
        """
        # 使用字符化排序避免 Python3 中 int/str 无法比较导致的报错
        return tuple(sorted(result, key=lambda row: str(row)))

    def select_best_query(self, sql_queries: List[str]) -> str:
        if not sql_queries:
            return "ERROR: no SQL queries provided"
            
        result_groups_valid = defaultdict(list)
        error_queries =[]
        
        for sql_query in sql_queries:
            exec_result = self._execute_sql(sql_query)
            if exec_result.result_type == SQLExecutionResultType.SUCCESS:
                if self._is_valid_execution_result(exec_result.result):
                    # 使用新的标准化 Key 进行分组
                    key = self._get_canonical_result_key(exec_result.result)
                    result_groups_valid[key].append(sql_query)
                else:
                    error_queries.append(sql_query)
            else:
                error_queries.append(sql_query)

        if not result_groups_valid:
            return sql_queries[0]

        total_queries = sum(len(v) for v in result_groups_valid.values())
        candidates =[]

        for key, sql_list in result_groups_valid.items():
            ex_score = len(sql_list) / total_queries if total_queries > 0 else 0
            exec_times = {sql: self._measure_execution_time(sql) for sql in sql_list}

            for sql in sql_list:
                time_penalty = exec_times[sql]
                final_score = self.alpha * ex_score - self.gamma * time_penalty
                candidates.append((sql, final_score, ex_score, time_penalty))

        for sql in error_queries:
            candidates.append((sql, -1.0, 0, float("inf")))

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_sql, best_score, ex, t = candidates[0]
        
        # 可选打印调试
        # print(f"\n[Best SQL] Final_Score={best_score:.3f} (EX_Ratio={ex:.3f}, Time_Penalty={t:.3f})")
        return best_sql


# ============================================================
# 3. 示例演示 - (可直接运行的测试用例)
# ============================================================

if __name__ == "__main__":
    db_path = "orchestra.sqlite"
    tables_json_path = "tables.json" # 虽然新函数不用，但为保持接口兼容而保留

    # --- 1. 创建临时的数据库和元数据文件 ---
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(tables_json_path):
        os.remove(tables_json_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE orchestra (
        Orchestra_ID INTEGER PRIMARY KEY,
        Orchestra TEXT,
        Record_Company TEXT,
        Year_of_Founded REAL,
        Major_Record_Format TEXT
    )
    """)
    # 插入数据，确保有符合条件和不符合条件的
    cursor.execute("INSERT INTO orchestra VALUES (1, 'London Symphony Orchestra', 'Decca', 1904, 'CD')")
    cursor.execute("INSERT INTO orchestra VALUES (2, 'New York Philharmonic', 'Sony Classical', 1842, 'CD')")
    cursor.execute("INSERT INTO orchestra VALUES (3, 'Berlin Philharmonic', 'Deutsche Grammophon', 1882, 'CD')")
    conn.commit()
    conn.close()

    # 创建一个虚拟的 tables.json 文件以满足函数接口
    with open(tables_json_path, 'w') as f:
        json.dump([{"db_id": "orchestra", "table_names_original": ["orchestra"], "column_names_original": [["*", "Orchestra_ID", "Orchestra", "Record_Company", "Year_of_Founded", "Major_Record_Format"]]}], f)

    # --- 2. 定义一组功能相同但风格各异的 SQL 查询 ---
    gold_sql_str = "SELECT Record_Company, Year_of_Founded FROM orchestra WHERE Year_of_Founded > 1850"

    pred_sql_newline = """
    SELECT
        Record_Company,
        Year_of_Founded
    FROM
        orchestra
    WHERE
        Year_of_Founded > 1850
    """

    pred_sql_single_line_nested = "SELECT T1.Record_Company, T1.Year_of_Founded FROM (SELECT * FROM orchestra WHERE Year_of_Founded > 1850) AS T1"

    pred_sql_cte = """
    WITH ModernOrchestras AS (
        SELECT Record_Company, Year_of_Founded
        FROM orchestra
        WHERE Year_of_Founded > 1850
    )
    SELECT * FROM ModernOrchestras
    """

    sql_variants =[
        gold_sql_str,                  
        pred_sql_newline,              
        pred_sql_single_line_nested,   
        pred_sql_cte,                  
        "SELECT Orchestra FROM orchestra WHERE Year_of_Founded < 1850", 
        "SELECT Record_Company FROM orchestra", 
        "SELECT Non_Existent_Column FROM orchestra", 
        "SELECT * FROM Non_Existent_Table",      
    ]

    print("=== 候选 SQL 列表 ===")
    for i, sql in enumerate(sql_variants):
        print(f"{i+1}: {sql.strip()}")

    # --- 添加的验证环节：Selector 也能顺带跑一下 ---
    selector = SelfConsistencySelectorV2(db_path=db_path, tables_json_path=tables_json_path)
    best_sql = selector.select_best_query(sql_variants)
    print("\n=== 最优 SQL 选择结果 ===")
    print(best_sql.strip())
    
    # --- 原生手动测试验证环节 ---
    print("\n=== 手动验证评估结果 ===")
    for name, pred_sql in {
        "Gold Standard": gold_sql_str,
        "Newline Variant": pred_sql_newline,
        "Single-line Nested Variant": pred_sql_single_line_nested,
        "CTE Variant": pred_sql_cte,
        "Wrong Result Variant": "SELECT Orchestra FROM orchestra WHERE Year_of_Founded < 1850",
        "Syntax Error Variant": "SELECT Non_Existent_Column FROM orchestra"
    }.items():
        # 这里不会再报 KeyError 了，因为字典键值修正为了 'exec'
        result = evaluate_single_sql(pred_sql, gold_sql_str, db_path, tables_json_path)
        print(f"[{name.ljust(28)}] -> EX Score: {result['exec']}")


    # --- 5. 清理临时文件 ---
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(tables_json_path):
        os.remove(tables_json_path)