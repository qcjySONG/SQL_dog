"""
主程序入口：用于测试和运行对话系统。
"""
import os
import sys
import json
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from agent import run_conversation
from database import get_ddl_dict, build_ddl_string
from config import DB_PATH, DDL_CACHE_DIR, MAX_TURNS


def test_database():
    """测试数据库连接和DDL生成"""
    print("=== 测试数据库连接 ===")
    if not os.path.exists(DB_PATH):
        print(f"错误：数据库文件不存在: {DB_PATH}")
        return False
    
    try:
        ddl_dict = get_ddl_dict(DB_PATH, DDL_CACHE_DIR)
        print(f"成功获取DDL，共 {len(ddl_dict)} 张表")
        for table in list(ddl_dict.keys())[:3]:
            print(f"  - {table}")
        return True
    except Exception as e:
        print(f"数据库测试失败: {e}")
        return False


def test_single_turn():
    """测试单轮对话"""
    print("\n=== 测试单轮对话 ===")
    question = "有多少患者？"
    print(f"问题: {question}")
    
    try:
        result = run_conversation(question)
        print("回答:")
        print(result["answer"])
        print(f"\n生成的SQL: {result['sql']}")
        print(f"执行成功: {result['success']}")
        return result
    except Exception as e:
        print(f"单轮对话测试失败: {e}")
        return None


def test_multi_turn():
    """测试多轮对话"""
    print("\n=== 测试多轮对话 ===")
    
    # 第一轮
    question1 = "有多少患者？"
    print(f"第一轮问题: {question1}")
    result1 = run_conversation(question1)
    print(f"SQL: {result1['sql']}")
    
    # 第二轮（使用历史）
    question2 = "其中男性患者有多少？"
    print(f"\n第二轮问题: {question2}")
    result2 = run_conversation(
        question2,
        history=result1["history"]
    )
    print(f"SQL: {result2['sql']}")
    print(f"执行成功: {result2['success']}")
    
    return result1, result2


def interactive_mode():
    """交互式模式"""
    print("\n=== 交互式模式 ===")
    print(f"最大对话轮次: {MAX_TURNS}")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'new' 开始新对话")
    print("输入 'history' 查看对话历史")
    
    history = []
    turn_count = 0
    conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    while turn_count < MAX_TURNS:
        try:
            question = input(f"\n[轮次 {turn_count + 1}/{MAX_TURNS}] 请输入问题: ").strip()
            
            if question.lower() in ['quit', 'exit']:
                break
            elif question.lower() == 'new':
                history = []
                turn_count = 0
                conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                print("已开始新对话")
                continue
            elif question.lower() == 'history':
                if not history:
                    print("对话历史为空")
                else:
                    print("对话历史:")
                    for i, turn in enumerate(history, 1):
                        print(f"  {i}. 问题: {turn['question']}")
                        print(f"     SQL: {turn['sql']}")
                continue
            elif not question:
                continue
            
            # 运行对话
            result = run_conversation(
                question=question,
                conversation_id=conversation_id,
                history=history
            )
            
            # 更新历史
            history = result["history"]
            turn_count += 1
            
            # 显示结果
            print(f"\nSQL: {result['sql']}")
            print(f"执行结果:\n{result['result']}")
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"处理出错: {e}")
    
    if turn_count >= MAX_TURNS:
        print(f"\n已达到最大对话轮次 ({MAX_TURNS})")


def main():
    """主函数"""
    print("SQL Dog - Text to SQL对话系统")
    print("=" * 50)
    
    # 测试数据库
    if not test_database():
        print("数据库测试失败，退出程序")
        return
    
    # 选择模式
    print("\n请选择运行模式:")
    print("1. 测试单轮对话")
    print("2. 测试多轮对话")
    print("3. 交互式对话")
    print("4. 启动Web界面")
    
    choice = input("请输入选择 (1-4): ").strip()
    
    if choice == '1':
        test_single_turn()
    elif choice == '2':
        test_multi_turn()
    elif choice == '3':
        interactive_mode()
    elif choice == '4':
        print("启动Web界面...")
        # 导入并运行前端
        from frontend.app import main as frontend_main
        frontend_main()
    else:
        print("无效选择")


if __name__ == "__main__":
    main()