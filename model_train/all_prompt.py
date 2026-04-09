def get_initPrompt(db_details, question, external_knowledge=None,Value=None):
    Prompt = f"""
# Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

# Database Engine:
SQLite

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
        Prompt+=f'''
# Value Retrieval
**Based on the current issues, we have obtained the following relatively similar retrieval values. Please use them appropriately when suitable:**
{Value}
'''

    Prompt += f"""
# Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
"""
    if Value:
        Prompt+=f"""
- The queries in the **[Value Retrieval]** section contain the database values most directly relevant to the current question. Please carefully assess their usefulness and use them appropriately!
"""
    Prompt+=f"""
- Before generating the final SQL query, please think carefully in <thinking> and </thinking> about the steps of how to write the query.

# Output Format:
<thinking>
Your thinking
</thinking>
```sql
-- Your SQL query
```
"""
    return Prompt



def get_initPrompt_EHR24(db_details, question, external_knowledge=None,Value=None,SQL_COT=None,ICL=None):
    Prompt = f"""
# Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

# Database Engine:
SQLite

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
    if ICL:
        Prompt+=f'''
# Example  
> We have provided some high-quality example data to assist you in answering the current user question.
{ICL}
'''

    if Value:
        Prompt+=f'''
# Value Retrieval
**Based on the current issues, we have obtained the following relatively similar retrieval values. Please use them appropriately when suitable:**
{Value}
'''
    if SQL_COT:
        Prompt+=f'''
# SQL Reasoning Chain
{SQL_COT}
'''

    Prompt += f"""
# Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
"""
    Prompt+=f"""
- Before generating the final SQL query, please think carefully in <thinking> and </thinking> about the steps of how to write the query.

# Output Format:
<thinking>
Your thinking
</thinking>
```sql
-- Your SQL query
```
"""
    return Prompt


def get_followupPrompt(db_details, question, history, external_knowledge=None,Value=None):
    Prompt = f"""
# Task Overview:
You are a data science expert continuing a conversation with a user. Below, you are provided with a database schema, the conversation history, and a new follow-up question. Your task is to generate a valid SQL query that correctly answers the new question, taking the context of the conversation into account.

# Database Engine:
SQLite

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
        Prompt+=f'''
# Value Retrieval
**Based on the current issues, we have obtained the following relatively similar retrieval values. Please use them appropriately when suitable:**
{Value}
'''
    Prompt += f"""
# Instructions:
- **Analyze the `Conversation History` and the `Follow-up Question` together.** The new question is likely related to the previous one.
- **Pay close attention to pronouns** (e.g., 'they', 'it'), comparative adjectives (e.g., 'more', 'cheaper', 'older'), and phrases that refer to the previous results (e.g., 'among those', 'from that list').
- You may need to **modify the previous SQL query** (e.g., by adding a condition to the WHERE clause, changing the ORDER BY, or altering the selected columns) to answer the new question.
- However, if the follow-up question is a completely new topic, generate a fresh query and do not rely on the history.
- **Specifically**, when the current question contains references such as "result1,result2...resultn" etc., these correspond to the execution results of "SQL1, SQL2...SQLn", etc.(n is the index of the previous dialogue turn.), from the conversation history. To answer the current question, you may need to reuse the corresponding SQL statements (e.g., via subqueries, CTEs, etc.).
"""
    if Value:
        Prompt+=f"""
- The queries in the **[Value Retrieval]** section contain the database values most directly relevant to the current question. Please carefully assess their usefulness and use them appropriately!
"""
    Prompt+=f"""
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think carefully in <thinking> and </thinking> about the steps of how to write the query.

# Output Format:
<thinking>
Your thinking
</thinking>
```sql
-- Your SQL query
```
"""
    return Prompt


def get_tablePrompt(db_details, question, history):
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
6. When producing your final answer, place your reasoning process between <thinking> and </thinking>.

# Output Format:
<thinking>
Your thinking
</thinking>
```json
{{
"table": [your selected tables]
}}
```
'''
    return Prompt


def OmniSQL_Prompt(db_details,question):
    input_prompt_template = f'''Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.'''
    return input_prompt_template