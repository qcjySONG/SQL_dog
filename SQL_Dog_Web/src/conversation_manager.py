"""
对话历史管理模块：负责对话历史的持久化存储和加载。
"""
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional


class ConversationManager:
    """对话历史管理器"""
    
    def __init__(self, storage_dir: str = "conversations"):
        """
        初始化对话历史管理器。
        
        Args:
            storage_dir: 对话历史存储目录
        """
        if storage_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            storage_dir = os.path.join(base_dir, conversations)
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def _get_conversation_file(self, conversation_id: str) -> str:
        """获取对话历史文件路径"""
        return os.path.join(self.storage_dir, f"{conversation_id}.json")
    
    def save_conversation(self, conversation_id: str, conversation_data: Dict) -> bool:
        """
        保存对话历史到文件。
        
        Args:
            conversation_id: 对话ID
            conversation_data: 对话数据
        
        Returns:
            是否保存成功
        """
        try:
            file_path = self._get_conversation_file(conversation_id)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存对话历史失败: {e}")
            return False
    
    def load_conversation(self, conversation_id: str) -> Optional[Dict]:
        """
        从文件加载对话历史。
        
        Args:
            conversation_id: 对话ID
        
        Returns:
            对话数据，如果不存在则返回None
        """
        try:
            file_path = self._get_conversation_file(conversation_id)
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载对话历史失败: {e}")
            return None
    
    def list_conversations(self) -> List[Dict]:
        """
        列出所有对话历史。
        
        Returns:
            对话历史列表，按创建时间倒序排列
        """
        conversations = []
        try:
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.json'):
                    conversation_id = filename[:-5]  # 移除.json后缀
                    conversation_data = self.load_conversation(conversation_id)
                    if conversation_data:
                        conversations.append({
                            'id': conversation_id,
                            'created_at': conversation_data.get('created_at', ''),
                            'turn_count': conversation_data.get('turn_count', 0),
                            'last_question': self._get_last_question(conversation_data)
                        })
            
            # 按创建时间倒序排列
            conversations.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return conversations
        except Exception as e:
            print(f"列出对话历史失败: {e}")
            return []
    
    def _get_last_question(self, conversation_data: Dict) -> str:
        """获取最后一个问题"""
        history = conversation_data.get('history', [])
        if history:
            return history[-1].get('question', '')[:50] + '...' if len(history[-1].get('question', '')) > 50 else history[-1].get('question', '')
        return ''
    
    def generate_conversation_summary(self, conversation_data: Dict) -> str:
        """
        使用LLM生成对话摘要。
        
        Args:
            conversation_data: 对话数据
        
        Returns:
            对话摘要（10字以内）
        """
        try:
            history = conversation_data.get('history', [])
            if not history:
                return "空对话"
            
            # 构建对话内容
            conversation_text = ""
            for i, turn in enumerate(history[:3]):  # 只取前3轮对话
                conversation_text += f"用户: {turn.get('question', '')}\n"
                if turn.get('sql'):
                    conversation_text += f"SQL: {turn['sql']}\n"
            
            # 使用LLM生成摘要
            from langchain_openai import ChatOpenAI
            import config
            
            llm = ChatOpenAI(
                model=config.DEEPSEEK_MODEL,
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_BASE_URL,
                temperature=0,
                max_tokens=50
            )
            
            prompt = f"""请为以下对话生成一个10字以内的中文摘要，概括对话的主要内容：

{conversation_text}

摘要："""
            
            response = llm.invoke([{"role": "user", "content": prompt}])
            summary = response.content.strip()
            
            # 确保摘要不超过10字
            if len(summary) > 10:
                summary = summary[:10]
            
            return summary
            
        except Exception as e:
            print(f"生成对话摘要失败: {e}")
            # 如果LLM失败，使用默认摘要
            history = conversation_data.get('history', [])
            if history:
                first_question = history[0].get('question', '')
                if len(first_question) > 8:
                    return first_question[:8] + ".."
                return first_question
            return "对话"
    
    def get_recent_conversations_with_summaries(self, max_count: int = 10) -> List[Dict]:
        """
        获取最近对话的摘要列表。
        
        Args:
            max_count: 最大返回数量
        
        Returns:
            包含摘要的对话列表
        """
        try:
            conversations = self.list_conversations()
            recent_conversations = []
            
            for conv in conversations[:max_count]:
                conversation_data = self.load_conversation(conv['id'])
                if conversation_data:
                    # 检查是否已有摘要
                    summary = conversation_data.get('summary', '')
                    if not summary:
                        # 生成新摘要
                        summary = self.generate_conversation_summary(conversation_data)
                        # 保存摘要
                        conversation_data['summary'] = summary
                        self.save_conversation(conv['id'], conversation_data)
                    
                    recent_conversations.append({
                        'id': conv['id'],
                        'summary': summary,
                        'turn_count': conv.get('turn_count', 0),
                        'created_at': conv.get('created_at', '')
                    })
            
            return recent_conversations
            
        except Exception as e:
            print(f"获取对话摘要列表失败: {e}")
            return []
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        删除对话历史。
        
        Args:
            conversation_id: 对话ID
        
        Returns:
            是否删除成功
        """
        try:
            file_path = self._get_conversation_file(conversation_id)
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"删除对话历史失败: {e}")
            return False
    
    def cleanup_old_conversations(self, max_count: int = 100) -> int:
        """
        清理旧的对话历史，保留最新的max_count个。
        
        Args:
            max_count: 最大保留数量
        
        Returns:
            删除的对话数量
        """
        try:
            conversations = self.list_conversations()
            if len(conversations) <= max_count:
                return 0
            
            # 删除多余的旧对话
            delete_count = 0
            for conversation in conversations[max_count:]:
                if self.delete_conversation(conversation['id']):
                    delete_count += 1
            
            return delete_count
        except Exception as e:
            print(f"清理对话历史失败: {e}")
            return 0



# 创建全局对话管理器实例
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
conversation_manager = ConversationManager(storage_dir=os.path.join(base_dir, "frontend", "conversations"))
