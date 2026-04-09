"""
Langgraph状态机：定义对话流程，支持闲聊和问数两种意图。
"""
import operator
import time
from typing import Annotated, Dict, List, Literal, TypedDict, Union, Callable, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

import config
from database import get_ddl_dict, execute_sql, format_sql_result
from prompts import get_initPrompt, get_followupPrompt, parse_sql_from_response, build_ddl_string
from table_selector import select_tables, filter_ddl_by_tables
from retrieval_router import run_retrieval_router

# 全局回调
_progress_callback: Optional[Callable] = None

def set_progress_callback(cb: Callable):
    global _progress_callback
    _progress_callback = cb

def emit_progress(step: str, data: dict = None):
    if data is None:
        data = {}
    data["timestamp"] = time.time()
    if _progress_callback:
        _progress_callback(step, data or {})


def extract_thinking_from_response(response: str) -> str:
    import re
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
    if thinking_match:
        return thinking_match.group(1).strip()
    return ""


def merge_step_details(existing: Dict, new: Dict) -> Dict:
    merged = dict(existing)
    merged.update(new)
    return merged


class AgentState(TypedDict):
    intent: str
    history: Annotated[List[Dict[str, str]], operator.add]
    current_question: str
    ddl: str
    generated_sql: str
    thinking_process: str
    sql_result: Union[str, List[Dict], None]
    error_message: str
    retry_count: int
    success: bool
    final_answer: str
    conversation_id: str
    selected_tables: List[str]
    ddl_dict: Dict[str, str]
    step_details: Dict[str, str]
    external_knowledge: str
    Value: str


def intent_classification_node(state: AgentState) -> Dict:
    print("--- 意图识别 ---")
    emit_progress("intent_start", {})
    question = state["current_question"]
    history = state.get("history", [])
    
    history_questions = []
    for turn in history[-5:]:
        if turn.get("question"):
            history_questions.append(turn["question"])
    
    history_context = "\n".join([f"- {q}" for q in history_questions]) if history_questions else "无历史对话"
    
    intent_prompt = f"""你是一个意图识别助手。请判断用户的当前问题是想进行闲聊还是查询数据库。

历史问题：
{history_context}

当前问题：{question}

请根据以下规则判断意图：
1. 如果用户在进行日常闲聊、问候、表达情感、询问与数据库无关的问题，返回 "chat"
2. 如果用户想查询数据库中的数据、统计信息、医疗记录、患者信息等，返回 "query"

只返回 "chat" 或 "query"，不要任何解释。
"""
    
    llm = ChatOpenAI(
        model=config.MODELSCOPE_MODEL,
        api_key=config.MODELSCOPE_API_KEY,
        base_url=config.MODELSCOPE_BASE_URL,
        temperature=0,
        max_tokens=10,
    )
    
    response = llm.invoke([HumanMessage(content=intent_prompt)])
    intent = response.content.strip().lower()
    
    if intent not in ["chat", "query"]:
        intent = "query"
    
    print(f"识别到的意图: {intent}")
    emit_progress("intent_done", {"intent": intent})
    return {"intent": intent, "step_details": {"intent": intent}}


def chat_node(state: AgentState) -> Dict:
    print("--- 闲聊模式 ---")
    question = state["current_question"]
    history = state.get("history", [])
    
    conversation_context = ""
    for turn in history[-10:]:
        conversation_context += f"用户: {turn.get('question', '')}\n"
        if turn.get("answer"):
            conversation_context += f"助手: {turn['answer']}\n"
    
    chat_prompt = f"""你是一个友好的AI助手。请根据对话历史和用户的当前问题，生成自然、友好的回复。

对话历史：
{conversation_context}

当前问题：{question}

请生成回复：
"""
    
    llm = ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0.7,
        max_tokens=1024,
    )
    
    response = llm.invoke([HumanMessage(content=chat_prompt)])
    answer = response.content
    
    new_history = {
        "question": question,
        "answer": answer,
        "intent": "chat"
    }
    
    return {
        "final_answer": answer,
        "history": [new_history],
    }


def generate_sql_node(state: AgentState) -> Dict:
    print("--- 生成SQL ---")
    emit_progress("sql_start", {})
    question = state["current_question"]
    ddl = state["ddl"]
    history = state.get("history", [])
    external_knowledge = state.get("external_knowledge") or None
    value_result = state.get("Value") or None
    
    if history is None or len(history) == 0:
        prompt = get_initPrompt(db_details=ddl, question=question, Database_engine="SQLite",
                                external_knowledge=external_knowledge, Value=value_result)
    else:
        history_str = ""
        for i, turn in enumerate(history, 1):
            history_str += f"<Question {i}>: {turn.get('question', '')}\n"
            if turn.get('sql'):
                history_str += f"<SQL {i}>: {turn['sql']}\n\n"
        
        prompt = get_followupPrompt(
            db_details=ddl,
            question=question,
            history=history_str,
            Database_engine="SQLite",
            external_knowledge=external_knowledge,
            Value=value_result
        )
    
    llm = ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0,
        max_tokens=1024,
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    sql_text = response.content
    
    thinking_process = extract_thinking_from_response(sql_text)
    generated_sql = parse_sql_from_response(sql_text)
    
    emit_progress("sql_done", {"sql": generated_sql})
    return {
        "generated_sql": generated_sql,
        "thinking_process": thinking_process
    }


def execute_sql_node(state: AgentState) -> Dict:
    print("--- 执行SQL ---")
    sql = state["generated_sql"]
    retry_count = state.get("retry_count", 0)
    
    print(f"执行SQL: {sql}")
    
    success, result = execute_sql(config.DB_PATH, sql)
    
    print(f"SQL执行结果: success={success}, result={result}")
    
    if success:
        formatted_result = format_sql_result(result)
        return {
            "sql_result": formatted_result,
            "success": True,
            "error_message": "",
        }
    else:
        print(f"SQL执行失败: {result}")
        return {
            "sql_result": None,
            "success": False,
            "error_message": result,
            "retry_count": retry_count + 1,
        }


def error_fix_node(state: AgentState) -> Dict:
    print("--- 错误修复 ---")
    question = state["current_question"]
    ddl = state["ddl"]
    history = state.get("history", [])
    error_message = state["error_message"]
    previous_sql = state["generated_sql"]
    
    error_question = f"""你之前的SQL执行出错了，请根据错误信息修正SQL。

用户原始问题：{question}

你之前的SQL：
```sql
{previous_sql}
```

错误信息：
{error_message}

请生成修正后的SQL查询。"""
    
    if history is None or len(history) == 0:
        prompt = get_initPrompt(db_details=ddl, question=error_question, Database_engine="SQLite")
    else:
        history_str = ""
        for i, turn in enumerate(history, 1):
            history_str += f"<Question {i}>: {turn.get('question', '')}\n"
            if turn.get('sql'):
                history_str += f"<SQL {i}>: {turn['sql']}\n\n"
        
        prompt = get_followupPrompt(
            db_details=ddl,
            question=error_question,
            history=history_str,
            Database_engine="SQLite"
        )
    
    llm = ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0,
        max_tokens=1024,
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    sql_text = response.content
    
    fixed_sql = parse_sql_from_response(sql_text)
    
    return {"generated_sql": fixed_sql}


def final_answer_node(state: AgentState) -> Dict:
    print("--- 生成最终答案 ---")
    question = state["current_question"]
    sql = state["generated_sql"]
    result = state["sql_result"]
    success = state["success"]
    error_message = state["error_message"]
    thinking_process = state.get("thinking_process", "")
    
    if success:
        result_markdown = format_result_to_markdown(result)
        
        final_answer_parts = []
        
        if thinking_process:
            final_answer_parts.append("**思考过程：**")
            final_answer_parts.append(f"<details><summary>点击展开思考过程</summary>\n{thinking_process}\n</details>")
            final_answer_parts.append("")
        
        final_answer_parts.append("**SQL查询：**")
        final_answer_parts.append(f"```sql\n{sql}\n```")
        final_answer_parts.append("")
        final_answer_parts.append("**执行结果：**")
        final_answer_parts.append(result_markdown)
        
        final_answer = "\n".join(final_answer_parts)
    else:
        final_answer_parts = []
        if thinking_process:
            final_answer_parts.append("**思考过程：**")
            final_answer_parts.append(f"<details><summary>点击展开思考过程</summary>\n{thinking_process}\n</details>")
            final_answer_parts.append("")
        final_answer_parts.append("**SQL查询：**")
        final_answer_parts.append(f"```sql\n{sql}\n```")
        final_answer_parts.append("")
        final_answer_parts.append("❌ SQL执行失败")
        final_answer_parts.append(f"**错误信息：** {error_message}")
        final_answer = "\n".join(final_answer_parts)
    
    new_history = {
        "question": question,
        "sql": sql,
        "result": {
            "success": success,
            "rows": result if isinstance(result, list) else [],
            "columns": list(result[0].keys()) if isinstance(result, list) and len(result) > 0 else [],
            "row_count": len(result) if isinstance(result, list) else 0,
            "error": error_message if not success else None
        },
        "error": error_message if not success else None
    }
    
    return {
        "final_answer": final_answer,
        "history": [new_history],
    }


def format_result_to_markdown(result: str) -> str:
    if not result or result == "查询结果为空":
        return "查询结果为空"
    
    lines = result.split('\n')
    if len(lines) < 3:
        return result
    
    header = lines[0]
    columns = header.split(' | ')
    
    html_lines = []
    html_lines.append('<table style="border-collapse: collapse; width: 100%;">')
    
    html_lines.append('<thead>')
    html_lines.append('<tr>')
    for col in columns:
        html_lines.append(f'<th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2; text-align: left;">{col.strip()}</th>')
    html_lines.append('</tr>')
    html_lines.append('</thead>')
    
    html_lines.append('<tbody>')
    for line in lines[2:]:
        if line.startswith('...'):
            html_lines.append('<tr>')
            html_lines.append(f'<td colspan="{len(columns)}" style="border: 1px solid #ddd; padding: 8px; text-align: center; font-style: italic;">{line}</td>')
            html_lines.append('</tr>')
        else:
            values = line.split(' | ')
            if len(values) == len(columns):
                html_lines.append('<tr>')
                for value in values:
                    html_lines.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{value.strip()}</td>')
                html_lines.append('</tr>')
    
    html_lines.append('</tbody>')
    html_lines.append('</table>')
    
    return '\n'.join(html_lines)


def should_retry(state: AgentState) -> Literal["error_fix", "final_answer"]:
    if state["success"]:
        return "final_answer"
    else:
        if state["retry_count"] < config.MAX_SQL_RETRIES:
            return "error_fix"
        else:
            return "final_answer"


def retrieval_router_node(state: AgentState) -> Dict:
    print("--- 检索路由 ---")
    emit_progress("retrieval_start", {})
    question = state["current_question"]
    ddl = state["ddl"]
    selected_tables = state.get("selected_tables", [])
    
    history = state.get("history", [])
    retrieval_result = run_retrieval_router(
        question=question,
        ddl=ddl,
        selected_tables=selected_tables,
        history=history
    )
    
    external_knowledge = retrieval_result.get("external_knowledge")
    value_result = retrieval_result.get("Value")
    
    emit_progress("retrieval_done", {
        "external_knowledge": external_knowledge,
        "Value": value_result
    })
    
    return {
        "external_knowledge": external_knowledge or "",
        "Value": value_result or ""
    }


def table_selection_node(state: AgentState) -> Dict:
    print("--- 表选择 ---")
    emit_progress("table_start", {})
    question = state["current_question"]
    history = state.get("history", [])
    ddl_dict = state["ddl_dict"]
    
    selected_tables = select_tables(ddl_dict, question, history)
    
    filtered_ddl = filter_ddl_by_tables(ddl_dict, selected_tables)
    
    print(f"选中 {len(selected_tables)} 张表，DDL从 {len(build_ddl_string(ddl_dict))} 字符缩减为 {len(filtered_ddl)} 字符")
    emit_progress("table_done", {"tables": selected_tables})
    
    existing_steps = state.get("step_details", {})
    new_steps = {"tables": selected_tables}
    merged_steps = merge_step_details(existing_steps, new_steps)
    
    return {
        "selected_tables": selected_tables,
        "ddl": filtered_ddl,
        "step_details": merged_steps
    }


def route_by_intent(state: AgentState) -> Literal["chat", "table_selection"]:
    intent = state.get("intent", "query")
    if intent == "chat":
        return "chat"
    else:
        return "table_selection"


def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("intent_classification", intent_classification_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("table_selection", table_selection_node)
    workflow.add_node("retrieval_router", retrieval_router_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("error_fix", error_fix_node)
    workflow.add_node("final_answer", final_answer_node)
    
    workflow.set_entry_point("intent_classification")
    
    workflow.add_conditional_edges(
        "intent_classification",
        route_by_intent,
        {
            "chat": "chat",
            "table_selection": "table_selection",
        },
    )
    
    workflow.add_edge("table_selection", "retrieval_router")
    workflow.add_edge("retrieval_router", "generate_sql")
    workflow.add_edge("chat", END)
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_conditional_edges(
        "execute_sql",
        should_retry,
        {
            "error_fix": "error_fix",
            "final_answer": "final_answer",
        },
    )
    workflow.add_edge("error_fix", "execute_sql")
    workflow.add_edge("final_answer", END)
    
    app = workflow.compile()
    return app


graph_app = build_graph()


def run_conversation(
    question: str,
    conversation_id: str = None,
    history: List[Dict[str, str]] = None,
    progress_callback: Callable = None,
) -> Dict:
    global _progress_callback
    _progress_callback = progress_callback
    
    if history is None:
        history = []
    
    ddl_dict = get_ddl_dict(
        config.DB_PATH, 
        config.DDL_CACHE_DIR,
        ddl_file=getattr(config, 'DDL_FILE', None)
    )
    ddl_string = build_ddl_string(ddl_dict)
    
    initial_state: AgentState = {
        "intent": "",
        "history": history,
        "current_question": question,
        "ddl": ddl_string,
        "ddl_dict": ddl_dict,
        "selected_tables": [],
        "generated_sql": "",
        "sql_result": None,
        "error_message": "",
        "retry_count": 0,
        "success": False,
        "final_answer": "",
        "conversation_id": conversation_id or "",
        "step_details": {},
        "thinking_process": "",
        "external_knowledge": "",
        "Value": "",
    }
    
    final_state = graph_app.invoke(initial_state)
    
    intent = final_state.get("intent", "query")
    
    if intent == "chat":
        return {
            "answer": final_state["final_answer"],
            "sql": None,
            "result": None,
            "success": True,
            "history": final_state["history"],
            "step_details": final_state.get("step_details", {}),
            "selected_tables": [],
        }
    else:
        return {
            "answer": final_state["final_answer"],
            "sql": final_state.get("generated_sql", ""),
            "result": final_state.get("sql_result"),
            "success": final_state.get("success", False),
            "history": final_state["history"],
            "step_details": final_state.get("step_details", {}),
            "selected_tables": final_state.get("selected_tables", []),
        }


if __name__ == "__main__":
    result = run_conversation("有多少患者？")
    print("最终答案：")
    print(result["answer"])
