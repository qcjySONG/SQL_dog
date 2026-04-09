# -*- coding: utf-8 -*-
"""
SQL Dog - 智能问数前端（带步骤耗时显示）
"""
import sys
import os
import uuid
import time
import threading
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gradio as gr
from agent import run_conversation, set_progress_callback
from conversation_manager import conversation_manager

conversations = {}
logs = []
progress_events = []
progress_lock = threading.Lock()

LOGO_PATH = "/amax/storage/nfs/qcjySONG/SQL_Dog/assets/sql_dog_logo.jpg"

def generate_conversation_id():
    return str(uuid.uuid4())

def get_conversation_history(conversation_id):
    if conversation_id not in conversations:
        conv_data = conversation_manager.load_conversation(conversation_id)
        if conv_data:
            conversations[conversation_id] = conv_data
        else:
            conversations[conversation_id] = {
                "history": [],
                "created_at": datetime.now().isoformat(),
                "turn_count": 0
            }
    return conversations[conversation_id]

def add_log(level, message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = "[%s] [%s] %s" % (timestamp, level, message)
    logs.append(log_entry)
    if len(logs) > 50:
        logs.pop(0)
    return "\n".join(logs)

def on_progress(step, data):
    with progress_lock:
        progress_events.append({"step": step, "data": data, "ts": time.time()})

set_progress_callback(on_progress)

db_schema = {
    "patients": ["subject_id", "gender", "anchor_age", "anchor_year"],
    "admissions": ["hadm_id", "subject_id", "admittime", "dischtime", "diagnosis"],
    "prescriptions": ["prescription_id", "hadm_id", "drug", "dose_val_rx", "route"]
}

custom_css = """
.log-terminal { background-color: #0f172a !important; color: #4ade80 !important; font-family: monospace; }
"""

def chatbot_response(message, history, conversation_id):
    if not message.strip():
        yield history, "", add_log("INFO", "空输入已忽略")
        return

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "正在识别意图..."})
    yield history, "", add_log("USER", "用户提问: " + message[:50])

    conv_data = get_conversation_history(conversation_id)

    with progress_lock:
        progress_events.clear()

    result_holder = {"result": None, "done": False, "error": None}
    t_start = time.time()
    timestamps = {"intent_done": None, "table_done": None, "sql_done": None}

    def run_in_thread():
        try:
            result_holder["result"] = run_conversation(
                question=message,
                conversation_id=conversation_id,
                history=conv_data["history"],
                progress_callback=on_progress
            )
            result_holder["done"] = True
        except Exception as e:
            result_holder["error"] = str(e)
            result_holder["done"] = True

    t = threading.Thread(target=run_in_thread)
    t.start()

    last_step = ""
    seen_steps = set()

    while not result_holder["done"]:
        with progress_lock:
            events = list(progress_events)
        for evt in events:
            current_step = evt["step"]
            current_data = evt["data"]

            if current_step not in seen_steps:
                seen_steps.add(current_step)
                ts = current_data.get("timestamp", evt["ts"])
                timestamps[current_step] = ts
                add_log("DEBUG", "step=%s ts=%.3f start=%.3f diff=%.3f" % (current_step, ts, t_start, ts - t_start))

                if current_step == "intent_done":
                    dur_s = round(ts - t_start, 1)
                    intent = current_data.get("intent", "query")
                    if intent == "chat":
                        display = "意图: chat (闲聊模式)"
                    else:
                        display = "意图: " + intent + " (" + str(dur_s) + "s)\n正在选择表..."
                    history[-1] = {"role": "assistant", "content": display}
                    yield history, "", add_log("INFO", "意图: " + intent)

                elif current_step == "table_done":
                    tables = current_data.get("tables", [])
                    tables_str = ", ".join(tables) if tables else "全部表"
                    intent_ts = timestamps.get("intent_done") or t_start
                    table_ts = ts
                    intent_s = round(intent_ts - t_start, 1)
                    table_s = round(table_ts - intent_ts, 1)
                    display = "意图: query (" + str(intent_s) + "s)\n表选择: " + tables_str + " (" + str(table_s) + "s)\n正在生成SQL..."
                    history[-1] = {"role": "assistant", "content": display}
                    yield history, "", add_log("INFO", "选中表: " + str(tables))

                elif current_step == "sql_done":
                    tables = current_data.get("tables", [])
                    tables_str = ", ".join(tables) if tables else "全部表"
                    intent_ts = timestamps.get("intent_done") or t_start
                    table_ts = timestamps.get("table_done") or intent_ts
                    sql_ts = ts
                    intent_s = round(intent_ts - t_start, 1)
                    table_s = round(table_ts - intent_ts, 1)
                    sql_s = round(sql_ts - table_ts, 1)
                    display = "意图: query (" + str(intent_s) + "s)\n表选择: " + tables_str + " (" + str(table_s) + "s)\nSQL生成完成 (" + str(sql_s) + "s)"
                    history[-1] = {"role": "assistant", "content": display}
                    yield history, "", add_log("INFO", "SQL生成完成")

        time.sleep(0.05)

    t.join()

    with progress_lock:
        remaining_events = list(progress_events)
    for evt in remaining_events:
        current_step = evt["step"]
        current_data = evt["data"]
        if current_step not in seen_steps:
            seen_steps.add(current_step)
            ts = current_data.get("timestamp", evt["ts"])
            timestamps[current_step] = ts
            add_log("DEBUG", "step=%s ts=%.3f start=%.3f diff=%.3f" % (current_step, ts, t_start, ts - t_start))

    if result_holder["error"]:
        history[-1] = {"role": "assistant", "content": "处理出错: " + result_holder["error"]}
        yield history, "", add_log("ERROR", "处理失败: " + result_holder["error"])
        return

    result = result_holder["result"]
    step_details = result.get("step_details", {})
    intent = step_details.get("intent", "query")
    tables = result.get("selected_tables", [])

    if intent == "chat":
        conv_data["history"] = result["history"]
        conv_data["turn_count"] += 1
        conversation_manager.save_conversation(conversation_id, conv_data)
        history[-1] = {"role": "assistant", "content": result["answer"]}
        yield history, "", add_log("SUCCESS", "闲聊回复")
        return

    conv_data["history"] = result["history"]
    conv_data["turn_count"] += 1
    conversation_manager.save_conversation(conversation_id, conv_data)

    intent_ts = timestamps.get("intent_done") or t_start
    table_ts = timestamps.get("table_done") or intent_ts
    sql_ts = timestamps.get("sql_done") or table_ts
    intent_s = round(intent_ts - t_start, 1)
    table_s = round(table_ts - intent_ts, 1)
    sql_s = round(sql_ts - table_ts, 1)
    total_s = round(sql_ts - t_start, 1)

    final_content = "<details><summary>执行步骤</summary>\n\n"
    final_content += "**意图识别**: " + intent + " — " + str(intent_s) + "s\n\n"
    if tables:
        final_content += "**表选择**: " + ", ".join(tables) + " — " + str(table_s) + "s\n\n"
    final_content += "**SQL生成**: " + str(sql_s) + "s\n\n"
    final_content += "**总耗时**: " + str(total_s) + "s\n\n"
    final_content += "</details>\n\n"
    final_content += result["answer"]

    history[-1] = {"role": "assistant", "content": final_content}
    add_log("SUCCESS", "查询完成")
    if result.get("sql"):
        add_log("SQL", "SQL: " + result['sql'][:100])

    yield history, "", add_log("INFO", "回复已生成")

def get_conversation_choices():
    try:
        recent_convs = conversation_manager.get_recent_conversations_with_summaries(15)
        choices = []
        for conv in recent_convs:
            display = conv['summary'] + " (" + str(conv['turn_count']) + "轮)"
            choices.append((display, conv['id']))
        return choices
    except:
        return []

def create_demo():
    with gr.Blocks(title="SQL Dog") as demo:
        conversation_id_state = gr.State(value=generate_conversation_id())

        gr.Markdown("## 🐕 SQL Dog - 智能问数助手")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 历史对话")
                history_radio = gr.Radio(choices=get_conversation_choices(), label="", interactive=True)
                new_chat_btn = gr.Button("+ 新建对话", variant="primary", size="sm")
            
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(show_label=False, height=500, render_markdown=True)
                with gr.Row():
                    msg_input = gr.Textbox(placeholder="输入你的问题，例如：统计各性别的患者数量", show_label=False, scale=8, container=False, lines=1)
                    send_btn = gr.Button("发送", variant="primary", scale=1)
            
            with gr.Column(scale=1):
                gr.JSON(label="数据库 Schema", value=db_schema, show_indices=False)
                log_output = gr.Textbox(elem_classes=["log-terminal"], label="执行日志", lines=15, interactive=False, value=add_log("SYSTEM", "SQL Dog 启动成功"))

        conversation_id_display = gr.Textbox(value=generate_conversation_id(), visible=False)

        def refresh_history():
            return gr.update(choices=get_conversation_choices(), value=None)

        def submit_message(message, history, conv_id, current_log):
            if not message.strip():
                yield history, "", conv_id, add_log("INFO", "空输入已忽略"), refresh_history()
                return
            for updated_history, _, new_log in chatbot_response(message, history, conv_id):
                yield updated_history, "", conv_id, new_log, refresh_history()

        msg_input.submit(submit_message, inputs=[msg_input, chatbot, conversation_id_display, log_output], outputs=[chatbot, msg_input, conversation_id_display, log_output, history_radio])
        send_btn.click(submit_message, inputs=[msg_input, chatbot, conversation_id_display, log_output], outputs=[chatbot, msg_input, conversation_id_display, log_output, history_radio])

        def handle_new_chat():
            new_id = generate_conversation_id()
            return [], new_id, add_log("INFO", "新建对话"), refresh_history()

        new_chat_btn.click(handle_new_chat, outputs=[chatbot, conversation_id_display, log_output, history_radio])

        def handle_select_conv(value):
            if value:
                conv_data = conversation_manager.load_conversation(value)
                if conv_data:
                    conversations[value] = conv_data
                    history = []
                    for turn in conv_data["history"]:
                        history.append({"role": "user", "content": turn["question"]})
                        if turn.get("intent") == "chat":
                            history.append({"role": "assistant", "content": turn.get("answer", "")})
                        else:
                            sql = turn.get("sql", "")
                            result = turn.get("result", {})
                            parts = []
                            if sql:
                                parts.append("**SQL查询：**\n```sql\n" + sql + "\n```")
                            if result and result.get("success"):
                                parts.append("**查询结果：** 共 " + str(result.get('row_count', 0)) + " 行数据")
                            history.append({"role": "assistant", "content": "\n\n".join(parts)})
                    return history, value, add_log("INFO", "历史对话已加载"), refresh_history()
            return [], "", add_log("WARN", "未找到选中的对话"), refresh_history()

        history_radio.change(handle_select_conv, inputs=[history_radio], outputs=[chatbot, conversation_id_display, log_output, history_radio])

        return demo

def main():
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=custom_css
    )

if __name__ == "__main__":
    main()
