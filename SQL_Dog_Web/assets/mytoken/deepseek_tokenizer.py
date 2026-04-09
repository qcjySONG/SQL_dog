import os
from transformers import AutoTokenizer


def truncate_text_by_tokens(text, max_tokens=4096):
    """
    截断文本使其 token 数不超过 max_tokens，并返回截断后的字符串。
    自动使用当前脚本目录下的 tokenizer。

    参数：
        text (str): 原始字符串
        max_tokens (int): 截断的最大 token 数

    返回：
        str: 截断后的字符串
    """
    # 获取当前 Python 脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chat_tokenizer_dir = current_dir  # tokenizer 路径为当前目录

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(chat_tokenizer_dir, trust_remote_code=False)

    # 使用 truncation=True 截断输入
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_tokens,
        return_tensors=None,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    # 解码回截断后的字符串
    truncated_text = tokenizer.decode(inputs["input_ids"], skip_special_tokens=True)
    return truncated_text


SL='''
We are given a problem and a previous SQL query that was executed. We need to analyze the SQL based on the given steps and the reference columns.
'''

def get_token_count(text: str) -> int:
    """
    计算文本的 token 数量。
    自动使用当前脚本目录下的 tokenizer。

    参数：
        text (str): 需要计算 token 数量的原始字符串。

    返回：
        int: 文本对应的 token 数量。
    """
    # 获取当前 Python 脚本所在目录
    # 注意: 在交互式环境（如 Jupyter）中，__file__ 可能未定义。
    # 在这种情况下，您可能需要手动指定路径或使用 os.getcwd()。
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # 如果在交互式环境中运行，__file__ 未定义，则使用当前工作目录
        current_dir = os.getcwd()
        
    tokenizer_dir = current_dir  # tokenizer 路径为当前目录

    # 加载 tokenizer
    # 确保 tokenizer 文件 (tokenizer.json, etc.) 存在于该目录
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=False)

    # 使用 tokenizer.encode() 对文本进行编码，它会返回一个 token ID 的列表
    token_ids = tokenizer.encode(text)

    # 列表的长度就是 token 的数量
    return len(token_ids)

if __name__ == "__main__":   
        print(truncate_text_by_tokens(text=SL))


