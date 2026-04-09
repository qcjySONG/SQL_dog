"""
提示词模板和构建函数。
"""
from typing import List, Dict, Tuple

def get_initPrompt(db_details, question, Database_engine="SQLite", external_knowledge=None, Value=None):
    """
    第一轮Prompt：生成初始SQL查询提示词。
    
    Args:
        db_details: 数据库DDL字符串
        question: 用户问题
        Database_engine: 数据库引擎（如 'SQLite'）
        external_knowledge: 外部知识（可选）
        Value: 值检索信息（可选）
    
    Returns:
        完整的提示词字符串。
    """
    Prompt = f"""
# Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

# Database Engine:
{Database_engine}

# Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.
"""
    
    if external_knowledge:
        Prompt += f"""
# External Knowledge:
{external_knowledge}
"""
    
    Prompt += f"""
# Question:
{question}
"""
    if Value:
        Prompt += f'''
# Value Retrieval
{Value}
'''

    Prompt += f"""
# Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think carefully in <thinking> and </thinking> about the steps of how to write the query.

# Output Format:
<thinking>
- **Analyze the User's Intent:**...
- **Consider Conversation History:**...
- **Map to Schema:**...
- **Formulate a Plan:**...
</thinking>
```sql
-- Your SQL query
```

# Question:
{question}
"""
    return Prompt

def get_followupPrompt(db_details, question, history, Database_engine="SQLite", external_knowledge=None, Value=None):
    """
    后续轮次Prompt：生成多轮对话SQL查询提示词。
    
    Args:
        db_details: 数据库DDL字符串
        question: 用户当前问题
        history: 历史对话字符串
        Database_engine: 数据库引擎（如 'SQLite'）
        external_knowledge: 外部知识（可选）
        Value: 值检索信息（可选）
    
    Returns:
        完整的提示词字符串。
    """
    Prompt = f"""
# Task Overview:
You are a data science expert continuing a conversation with a user. Below, you are provided with a database schema, the conversation history, and a new follow-up question. Your task is to generate a valid SQL query that correctly answers the new question, taking the context of the conversation into account.

# Database Engine:
{Database_engine}

# Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, and foreign keys.
"""
    
    if external_knowledge:
        Prompt += f"""
# External Knowledge:
{external_knowledge}
"""
    
    Prompt += f"""
# Conversation History:
{history}

* This section provides the user's previous questions along with their corresponding SQL queries. Please use this context to understand the user's specific intent.

# Follow-up Question:
{question}
"""
    if Value:
        Prompt += f'''
# Value Retrieval
{Value}
'''
    Prompt += f"""
# Instructions:
- **Analyze the `Conversation History` and the `Follow-up Question` together.** The new question is likely related to the previous one.
- **Pay close attention to pronouns** (e.g., 'they', 'it'), comparative adjectives (e.g., 'more', 'cheaper', 'older'), and phrases that refer to the previous results (e.g., 'among those', 'from that list').
- You may need to **modify the previous SQL query** (e.g., by adding a condition to the WHERE clause, changing the ORDER BY, or altering the selected columns) to answer the new question.
- However, if the follow-up question is a completely new topic, generate a fresh query and do not rely on the history.
- **Specifically**, when the current question contains references such as "result1,result2...resultn" etc., these correspond to the execution results of "SQL1, SQL2...SQLn", etc.(n is the index of the previous dialogue turn.), from the conversation history. To answer the current question, you may need to reuse the corresponding SQL statements (e.g., via subqueries, CTEs, etc.).
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think carefully in <thinking> and </thinking> about the steps of how to write the query.

# Output Format:
<thinking>
- **Analyze the User's Intent:**...
- **Consider Conversation History:**...
- **Map to Schema:**...
- **Formulate a Plan:**...
</thinking>
```sql
-- Your SQL query
```

# Follow-up Question:
{question}
"""
    return Prompt


def build_ddl_string(ddl_dict: Dict[str, str]) -> str:
    """
    将DDL字典拼接成一个字符串，用于提示词。
    
    Args:
        ddl_dict: DDL字典，键为表名，值为CREATE TABLE语句。
    
    Returns:
        拼接后的DDL字符串。
    """
    ddl_parts = []
    for table, ddl in ddl_dict.items():
        ddl_parts.append(ddl)
    return "\n\n".join(ddl_parts)


def parse_sql_from_response(response: str) -> str:
    """
    从LLM响应中提取SQL语句。
    
    Args:
        response: LLM的响应文本。
    
    Returns:
        清理后的SQL语句。
    """
    import re
    
    # 尝试从markdown代码块中提取SQL
    # 匹配 ```sql ... ``` 格式
    sql_match = re.search(r'```sql\s*\n(.*?)\n```', response, re.DOTALL)
    if sql_match:
        sql = sql_match.group(1).strip()
    else:
        # 尝试匹配 ``` ... ``` 格式
        code_match = re.search(r'```\s*\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            sql = code_match.group(1).strip()
        else:
            # 如果没有代码块，尝试找到SQL关键字开始的部分
            # 查找SELECT、WITH、INSERT、UPDATE、DELETE等SQL关键字
            sql_keywords = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']
            sql = response.strip()
            
            # 查找第一个SQL关键字的位置
            start_idx = len(sql)
            for keyword in sql_keywords:
                idx = sql.upper().find(keyword)
                if idx != -1 and idx < start_idx:
                    start_idx = idx
            
            if start_idx < len(sql):
                sql = sql[start_idx:]
    
    # 去除末尾的分号（可选，根据需求）
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    
    return sql


def get_tablePrompt_d(db_details, question, history):
    Prompt = f'''
# Role
You are a data architect proficient in database schema analysis. Your task is to perform table selection as part of schema linking.

# Task
I will provide you with the DDL of a SQLite database (including table names, column names, and foreign key relationships), a conversation history between the user and the database (History), and the current user question (Current Question).
Your job is to analyze the current question in the context of the conversation history and select the list of table names from the given database that are **strictly necessary** to answer this question.

# Database Schema:
{db_details}

# Conversation History:
{history}

# Current Question:
{question}

# Constraints
1. **Multi-turn dependency analysis**: Carefully review the conversation history. If the current question contains pronouns (e.g., "they", "these users"), resolve their referents based on the history. If the current question further filters or refines the results from the previous turn (e.g., "sort by date", "show their names"), you must retain the relevant tables used in the previous turn.
2. **Focus only on table selection**: Do not generate SQL queries—only output table names.
3. **Existence check**: Only select tables that are explicitly present in the [Database Schema]. Do not infer or assume additional tables.
4. When the problem contains descriptions such as "Result x" or similar, it represents the execution result of the corresponding SQL in round x. Please reuse the corresponding SQL!
5. **Minimalism principle**: Select only the tables that might be needed to answer the current question! If none of the provided tables can possibly answer the user's question, return an empty list.
6. Based on the current issue, we have obtained the following relatively similar retrieval values. Please use them correctly under appropriate circumstances.

# Output Format:
```json
{{
"table": [your selected tables] or []
}}
```
'''
    return Prompt


if __name__ == "__main__":
    # 测试
    ddl_example = {
        "patients": "CREATE TABLE patients (\n    subject_id INTEGER, -- 患者ID\n    gender TEXT -- 性别\n);",
        "admissions": "CREATE TABLE admissions (\n    subject_id INTEGER,\n    hadm_id INTEGER -- 住院ID\n);"
    }
    
    print("=== 单轮提示词 ===")
    prompt_single = build_prompt("查询所有男性患者", build_ddl_string(ddl_example))
    print(prompt_single)
    
    print("\n=== 伪多轮提示词 ===")
    history_example = [
        {"question": "有多少患者？", "sql": "SELECT COUNT(*) FROM patients"},
        {"question": "男性患者有多少？", "sql": "SELECT COUNT(*) FROM patients WHERE gender='M'"}
    ]
    prompt_multi = build_prompt("女性患者有多少？", build_ddl_string(ddl_example), history_example)
    print(prompt_multi)