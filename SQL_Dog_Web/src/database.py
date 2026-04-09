"""
数据库操作模块：包括DDL生成、缓存、SQL执行等。
"""
import sqlite3
import json
import os
import hashlib
from typing import Dict, List, Optional, Tuple


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


def get_db_hash(db_path: str) -> str:
    """计算数据库文件的哈希值，用于检测变化"""
    if not os.path.exists(db_path):
        return ""
    with open(db_path, 'rb') as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def load_ddl_cache(cache_path: str, db_path: str) -> Optional[Dict[str, str]]:
    """从缓存文件加载DDL，如果缓存有效"""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        cached_hash = cache_data.get('db_hash', '')
        current_hash = get_db_hash(db_path)
        if cached_hash != current_hash:
            return None
        return cache_data.get('ddl_map', {})
    except Exception:
        return None


def save_ddl_cache(cache_path: str, db_path: str, ddl_map: Dict[str, str]):
    """将DDL保存到缓存文件"""
    cache_data = {
        'db_hash': get_db_hash(db_path),
        'ddl_map': ddl_map
    }
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


def get_ddl_dict(db_path: str, cache_dir: str = "ddl_cache", schema_to_nl_mapping: Dict = None, ddl_file: str = None) -> Dict[str, str]:
    """
    获取数据库的DDL字典，优先使用DDL文件，其次使用缓存，最后从数据库生成。
    
    Args:
        db_path: 数据库文件路径
        cache_dir: 缓存目录
        schema_to_nl_mapping: 列名到自然语言描述的映射字典，键格式为"table.column"
        ddl_file: 直接指定DDL文件路径（如果提供，优先使用此文件）
    
    Returns:
        DDL字典，键为表名（小写），值为CREATE TABLE语句字符串
    """
    if schema_to_nl_mapping is None:
        schema_to_nl_mapping = {}
    
    # 如果指定了DDL文件，优先加载
    if ddl_file and os.path.exists(ddl_file):
        try:
            with open(ddl_file, 'r', encoding='utf-8') as f:
                ddl_data = json.load(f)
            # 兼容两种格式：直接是ddl_map或包含db_hash的缓存格式
            if 'ddl_map' in ddl_data:
                print(f"从DDL文件加载: {ddl_file}")
                return ddl_data['ddl_map']
            else:
                print(f"从DDL文件加载(直接格式): {ddl_file}")
                return ddl_data
        except Exception as e:
            print(f"加载DDL文件失败: {e}，尝试使用缓存...")
    
    os.makedirs(cache_dir, exist_ok=True)
    db_name = os.path.basename(db_path).replace('.sqlite', '').replace('.db', '')
    cache_path = os.path.join(cache_dir, f"{db_name}_ddl_cache.json")
    
    # 尝试加载缓存
    cached_ddl = load_ddl_cache(cache_path, db_path)
    if cached_ddl is not None:
        print(f"从缓存加载DDL: {cache_path}")
        return cached_ddl
    
    # 生成新的DDL
    print(f"生成新的DDL，数据库: {db_path}")
    ddl_map = generate_ddl_dict(db_path, schema_to_nl_mapping)
    save_ddl_cache(cache_path, db_path, ddl_map)
    print(f"DDL已缓存到: {cache_path}")
    return ddl_map


def execute_sql(db_path: str, sql: str) -> Tuple[bool, any]:
    """
    执行SQL语句。
    
    Args:
        db_path: 数据库文件路径
        sql: SQL语句
    
    Returns:
        (success, result)
        如果success为True，result为查询结果列表（每行为字典）或影响行数
        如果success为False，result为错误信息字符串
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        
        if sql.strip().upper().startswith(('SELECT', 'WITH', 'PRAGMA')):
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]
            return True, result
        else:
            conn.commit()
            return True, f"影响行数: {cursor.rowcount}"
    except sqlite3.Error as e:
        return False, str(e)
    finally:
        if conn:
            conn.close()


def format_sql_result(result: any, max_rows: int = 10) -> str:
    """格式化SQL执行结果为表格形式"""
    if isinstance(result, str):
        return result
    if not result:
        return "查询结果为空"
    
    # 获取列名
    if isinstance(result, list) and len(result) > 0:
        columns = list(result[0].keys())
    else:
        return str(result)
    
    # 计算每列的最大宽度
    col_widths = {}
    for col in columns:
        # 列名宽度
        col_widths[col] = len(str(col))
        # 数据宽度
        for row in result[:max_rows]:
            val_str = str(row.get(col, ''))
            col_widths[col] = max(col_widths[col], len(val_str))
    
    # 构建表格
    table_lines = []
    
    # 表头
    header = " | ".join(str(col).ljust(col_widths[col]) for col in columns)
    table_lines.append(header)
    
    # 分隔线
    separator = "-+-".join("-" * col_widths[col] for col in columns)
    table_lines.append(separator)
    
    # 数据行
    display_rows = result[:max_rows]
    for row in display_rows:
        row_str = " | ".join(str(row.get(col, '')).ljust(col_widths[col]) for col in columns)
        table_lines.append(row_str)
    
    # 添加行数信息
    if len(result) > max_rows:
        table_lines.append(f"... (共{len(result)}行，显示前{max_rows}行)")
    
    return "\n".join(table_lines)


if __name__ == "__main__":
    # 简单测试
    import sys
    db_path = "../DB/mimic_iv.sqlite"
    if os.path.exists(db_path):
        ddl = get_ddl_dict(db_path)
        print(f"共 {len(ddl)} 张表")
        for table, create_stmt in list(ddl.items())[:2]:
            print(f"\n--- {table} ---")
            print(create_stmt[:500])
    else:
        print(f"数据库文件不存在: {db_path}")