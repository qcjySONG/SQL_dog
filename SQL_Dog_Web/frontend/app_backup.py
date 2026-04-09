"""
Gradio前端界面：美观的ChatGPT风格设计
"""
import sys
import os
import uuid
import base64
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gradio as gr
from agent import run_conversation
from database import get_ddl_dict
from config import DB_PATH, DDL_CACHE_DIR, MAX_TURNS
from conversation_manager import conversation_manager

# 读取logo文件并转换为base64
LOGO_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'sql_dog_logo.jpg')

def get_logo_base64():
    """将logo图片转换为base64编码"""
    try:
        with open(LOGO_PATH, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
            return f"data:image/jpeg;base64,{logo_data}"
    except:
        return ""

LOGO_BASE64 = get_logo_base64()

# 对话管理
conversations = {}

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

def format_chat_message(role, content):
    """格式化聊天消息"""
    if role == "user":
        return {
            "role": "user",
            "content": content
        }
    else:
        return {
            "role": "assistant", 
            "content": content
        }

def chatbot_response(message, history, conversation_id):
    """处理用户输入并生成回复"""
    if not message.strip():
        yield history, ""
        return
    
    # 添加用户消息到历史
    history.append({"role": "user", "content": message})
    # 添加空的助手消息占位
    history.append({"role": "assistant", "content": ""})
    yield history, ""
    
    conv_data = get_conversation_history(conversation_id)
    
    if conv_data["turn_count"] >= MAX_TURNS:
        history[-1] = {"role": "assistant", "content": "⚠️ 对话轮次已达上限，请开始新的对话。"}
        yield history, ""
        return
    
    try:
        result = run_conversation(
            question=message,
            conversation_id=conversation_id,
            history=conv_data["history"]
        )
        
        conv_data["history"] = result["history"]
        conv_data["turn_count"] += 1
        conversation_manager.save_conversation(conversation_id, conv_data)
        
        history[-1] = {"role": "assistant", "content": result["answer"]}
        yield history, ""
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        history[-1] = {"role": "assistant", "content": f"❌ 处理出错：{str(e)}"}
        yield history, ""

def new_conversation():
    """创建新对话"""
    new_id = generate_conversation_id()
    return [], new_id

def load_conversation(conversation_id):
    """加载历史对话"""
    conv_data = conversation_manager.load_conversation(conversation_id)
    if conv_data:
        conversations[conversation_id] = conv_data
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
                    parts.append(f"**SQL查询：**\n```sql\n{sql}\n```")
                if result and result.get("success"):
                    parts.append(f"**查询结果：** 共 {result.get('row_count', 0)} 行数据")
                elif result and result.get("error"):
                    parts.append(f"**错误信息：** {result['error']}")
                history.append({"role": "assistant", "content": "\n\n".join(parts)})
        return history, conversation_id
    return [], ""

def get_conversation_choices():
    """获取对话列表"""
    try:
        recent_convs = conversation_manager.get_recent_conversations_with_summaries(20)
        choices = []
        for conv in recent_convs:
            display = f"💬 {conv['summary']} ({conv['turn_count']}轮)"
            choices.append((display, conv['id']))
        return choices
    except:
        return []

# CSS样式设计
custom_css = """
/* 全局容器 */
.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    background-color: #f8fafc !important;
    min-height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* 隐藏默认页脚 */
footer.svelte-zhy16l, footer {
    display: none !important;
}

/* 左侧边栏容器样式 */
div:has(> .sidebar-column) {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    min-width: 280px !important;
    width: 280px !important;
    border-right: 1px solid #334155;
    box-shadow: 2px 0 10px rgba(0,0,0,0.1);
}

/* 侧边栏内部元素 */
.sidebar-column * {
    color: #f1f5f9 !important;
}

/* 新对话按钮 */
.new-chat-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 20px !important;
    margin: 16px 20px !important;
    width: calc(100% - 40px) !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
}

.new-chat-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
}

/* 历史对话标题 */
.history-title {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    padding: 12px 24px !important;
    margin: 0 !important;
}

/* 历史下拉选择 */
.sidebar-dropdown {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    margin: 0 20px 12px 20px !important;
    width: calc(100% - 40px) !important;
}

.sidebar-dropdown:hover {
    border-color: #3b82f6 !important;
}

/* 加载按钮 */
.load-btn {
    background: rgba(59, 130, 246, 0.2) !important;
    color: #93c5fd !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
    border-radius: 10px !important;
    margin: 0 20px !important;
    width: calc(100% - 40px) !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.load-btn:hover {
    background: rgba(59, 130, 246, 0.3) !important;
    color: #bfdbfe !important;
}

/* 状态指示器 */
.status-indicator {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 16px 24px;
    font-size: 13px;
    color: #94a3b8 !important;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #22c55e;
}

/* 主聊天区域 */
.chatbot {
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%) !important;
    border: none !important;
    padding: 24px 40px !important;
}

/* 用户消息气泡 */
.chatbot .user {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
    color: white !important;
    border-radius: 18px 18px 6px 18px !important;
    padding: 14px 20px !important;
    margin: 12px 0 !important;
    max-width: 75% !important;
    margin-left: auto !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
    border: none !important;
}

/* 助手消息气泡 */
.chatbot .assistant {
    background: #ffffff !important;
    color: #1e293b !important;
    border-radius: 18px 18px 18px 6px !important;
    padding: 16px 22px !important;
    margin: 12px 0 !important;
    max-width: 85% !important;
    margin-right: auto !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08) !important;
    border: 1px solid #e2e8f0 !important;
}

/* 代码块样式 */
.chatbot .assistant pre {
    background: #0f172a !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    margin: 12px 0 !important;
    border: 1px solid #334155 !important;
}

.chatbot .assistant code {
    color: #e2e8f0 !important;
    font-family: 'Fira Code', Consolas, monospace !important;
}

/* 输入框样式 */
.input-box {
    background: #ffffff !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 20px !important;
    padding: 16px 24px !important;
    font-size: 15px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04) !important;
    transition: all 0.3s ease !important;
}

.input-box:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1) !important;
    outline: none !important;
}

/* 发送按钮 */
.send-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
    border-radius: 50% !important;
    width: 56px !important;
    height: 56px !important;
    min-width: 56px !important;
    border: none !important;
    color: white !important;
    font-size: 22px !important;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.35) !important;
    transition: all 0.3s ease !important;
}

.send-btn:hover {
    transform: scale(1.08) !important;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.45) !important;
}

/* 欢迎页面 */
.welcome-container {
    padding: 60px 40px !important;
}

.welcome-title {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-size: 36px !important;
    font-weight: 800 !important;
}

.suggestion-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}

.suggestion-card:hover {
    border-color: #3b82f6 !important;
    transform: translateY(-4px) !important;
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.15) !important;
}
"""

# 构建侧边栏HTML
def get_sidebar_html():
    logo_html = ""
    if LOGO_BASE64:
        logo_html = f'<img src="{LOGO_BASE64}" class="sidebar-logo" alt="Logo">'
    
    return f"""
    <div class="sidebar">
        <div class="sidebar-header">
            {logo_html}
            <span class="sidebar-title">SQL Dog</span>
        </div>
        <div class="status-indicator">
            <span class="status-dot"></span>
            <span>AI 已就绪</span>
        </div>
    </div>
    """

# 构建欢迎页面HTML
def get_welcome_html():
    logo_html = ""
    if LOGO_BASE64:
        logo_html = f'<img src="{LOGO_BASE64}" class="welcome-logo" alt="Logo">'
    
    return f"""
    <div class="welcome-container">
        {logo_html}
        <h1 class="welcome-title">🐕 SQL Dog</h1>
        <p class="welcome-subtitle">您的智能 SQL 查询助手 - 用自然语言与数据库对话</p>
        <div class="suggestion-cards">
            <div class="suggestion-card" onclick="document.querySelector('textarea').focus()">
                <div class="suggestion-icon">🔍</div>
                <div class="suggestion-title">查询患者信息</div>
                <div class="suggestion-desc">"有多少患者？"</div>
            </div>
            <div class="suggestion-card" onclick="document.querySelector('textarea').focus()">
                <div class="suggestion-icon">📊</div>
                <div class="suggestion-title">统计分析</div>
                <div class="suggestion-desc">"按性别统计患者数量"</div>
            </div>
            <div class="suggestion-card" onclick="document.querySelector('textarea').focus()">
                <div class="suggestion-icon">💊</div>
                <div class="suggestion-title">用药查询</div>
                <div class="suggestion-desc">"查询常用药物列表"</div>
            </div>
            <div class="suggestion-card" onclick="document.querySelector('textarea').focus()">
                <div class="suggestion-icon">❤️</div>
                <div class="suggestion-title">健康咨询</div>
                <div class="suggestion-desc">"如何保持健康？"</div>
            </div>
        </div>
    </div>
    """


def create_demo():
    """创建Gradio演示界面"""
    
    with gr.Blocks(
        title="SQL Dog - 智能SQL助手",
        fill_height=True
    ) as demo:
        # 存储状态
        conversation_id_state = gr.State(value=generate_conversation_id())
        
        # 主布局 - 使用Row分两栏
        with gr.Row(equal_height=True):
            # 左侧边栏
            with gr.Column(scale=0, min_width=260):
                # 侧边栏头部
                if LOGO_BASE64:
                    gr.HTML(f"""
                    <div style="padding: 20px; border-bottom: 1px solid rgba(255,255,255,0.1); display: flex; align-items: center; gap: 12px;">
                        <img src="{LOGO_BASE64}" style="width: 40px; height: 40px; border-radius: 8px; object-fit: cover;">
                        <span style="color: #ffffff; font-size: 18px; font-weight: 600;">SQL Dog</span>
                    </div>
                    """)
                else:
                    gr.HTML("""
                    <div style="padding: 20px; border-bottom: 1px solid rgba(255,255,255,0.1); display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 32px;">🐕</span>
                        <span style="color: #ffffff; font-size: 18px; font-weight: 600;">SQL Dog</span>
                    </div>
                    """)
                
                # 状态指示
                gr.HTML("""
                <div style="display: flex; align-items: center; gap: 8px; padding: 12px 20px; font-size: 13px; color: #8b8b9e;">
                    <span style="width: 8px; height: 8px; border-radius: 50%; background: #4ade80; display: inline-block;"></span>
                    <span>AI 已就绪</span>
                </div>
                """)
                
                # 新对话按钮
                new_chat_btn = gr.Button(
                    "+ 新对话",
                    variant="secondary",
                    size="sm",
                    elem_classes=["new-chat-btn"]
                )
                gr.HTML("<style>.new-chat-btn { width: calc(100% - 32px); margin: 0 16px 16px 16px; background: rgba(255,255,255,0.1); color: #fff; border: none; border-radius: 8px; padding: 12px 16px; cursor: pointer; text-align: left; font-size: 14px; } .new-chat-btn:hover { background: rgba(255,255,255,0.2); }</style>")
                
                # 历史对话标题
                gr.HTML("<div style='padding: 8px 20px; color: #8b8b9e; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>历史对话</div>")
                
                # 历史下拉选择
                history_dropdown = gr.Dropdown(
                    choices=get_conversation_choices(),
                    label="",
                    interactive=True,
                    elem_classes=["sidebar-dropdown"]
                )
                
                # 加载按钮
                load_btn = gr.Button("📂 加载选中对话", variant="secondary", size="sm")
            
            # 右侧主内容区
            with gr.Column():
                # 顶部栏
                gr.HTML("""
                <div style="padding: 16px 24px; border-bottom: 1px solid #e8e8e8; display: flex; align-items: center; justify-content: center;">
                    <h2 style="font-size: 16px; font-weight: 500; color: #1a1a1a; margin: 0;">SQL Dog - 智能SQL助手</h2>
                </div>
                """)
                
                # 聊天机器人
                chatbot = gr.Chatbot(
                    height=600,
                    render_markdown=True,
                    show_label=False,
                    elem_classes=["chatbot"]
                )
                
                # 欢迎页面
                welcome_html = get_welcome_html()
                welcome_display = gr.HTML(welcome_html)
                
                # 输入区域
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入你的问题，例如：有多少患者？",
                        lines=2,
                        show_label=False,
                        max_lines=8,
                        scale=1,
                        elem_classes=["input-box"]
                    )
                    send_btn = gr.Button(
                        "➤",
                        variant="primary",
                        scale=0,
                        min_width=56,
                        elem_classes=["send-btn"]
                    )
                
                # 隐藏的对话ID
                conversation_id_display = gr.Textbox(value=generate_conversation_id(), visible=False)
        
        # 事件处理
        def submit_message(message, history, conv_id):
            """处理消息提交"""
            if not message.strip():
                return history, "", gr.update(visible=True)
            
            # 隐藏欢迎页面
            for updated_history, _ in chatbot_response(message, history, conv_id):
                yield updated_history, "", gr.update(visible=False)
        
        # 发送按钮点击和回车提交
        msg_input.submit(
            submit_message,
            inputs=[msg_input, chatbot, conversation_id_display],
            outputs=[chatbot, msg_input, welcome_display]
        )
        
        send_btn.click(
            submit_message,
            inputs=[msg_input, chatbot, conversation_id_display],
            outputs=[chatbot, msg_input, welcome_display]
        )
        
        # 新对话按钮点击
        def handle_new_chat():
            return [], generate_conversation_id(), gr.update(visible=True)
        
        new_chat_btn.click(
            handle_new_chat,
            outputs=[chatbot, conversation_id_display, welcome_display]
        )
        
        # 加载历史对话
        def handle_load_conv(value):
            if value:
                history, conv_id = load_conversation(value)
                return history, conv_id, gr.update(visible=False)
            return [], "", gr.update(visible=True)
        
        load_btn.click(
            handle_load_conv,
            inputs=[history_dropdown],
            outputs=[chatbot, conversation_id_display, welcome_display]
        )
        
        # 下拉框变化自动加载
        history_dropdown.change(
            handle_load_conv,
            inputs=[history_dropdown],
            outputs=[chatbot, conversation_id_display, welcome_display]
        )
        
        return demo

def main():
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="gray",
        ),
        css=custom_css,
        inbrowser=False,
    )

if __name__ == "__main__":
    main()