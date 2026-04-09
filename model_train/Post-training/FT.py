import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, concatenate_datasets
from trl import SFTConfig, SFTTrainer
from transformers import TextStreamer, AutoTokenizer
import numpy as np
from transformers import TrainerCallback
from modelscope.msdatasets import MsDataset
from datasets import concatenate_datasets


# 0. 环境配置
# ==============================================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # export CUDA_VISIBLE_DEVICES=0



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# 定义路径变量，方便管理
model_path = os.path.join(BASE_DIR, "MODEL", "Qwne3_4B")
# Lora
output_dir = os.path.join(SCRIPT_DIR, "lora", "4B_Pro")
merged_output_dir = os.path.join(SCRIPT_DIR, "model", "4B_Pro")
# Lora 参数
Lora_R=32

mode="CoT"

epochs=2

max_seq_length=10000

def extract_question_content(text):
    """
    提取字符串中从 '# Question:' 或 '# Follow-up Question:' 
    到第一个 '\n\n' 之间的内容（包含标记本身）
    
    参数:
        text (str): 输入文本
        
    返回:
        str: 提取的完整内容（包含标记），如果未找到则返回空字符串
    """
    # 支持的标记列表
    markers = ["# Question:", "# Follow-up Question:"]
    
    for marker in markers:
        start_index = text.find(marker)
        if start_index != -1:
            # 从标记开始位置查找结束标记 \n\n
            end_index = text.find("\# Constraints", start_index)
            if end_index == -1:
                # 如果没有找到 \n\n，则提取到文本末尾
                extracted = text[start_index:].strip()
            else:
                extracted = text[start_index:end_index].strip()
            return extracted
    
    return ""

class DebugLossMaskCallback(TrainerCallback):
    def __init__(self):
        self.printed = False

    def on_train_batch_begin(self, args, state, control, **kwargs):
        if self.printed:
            return

        inputs = kwargs.get("inputs", None)
        if inputs is None or "labels" not in inputs:
            return

        labels = inputs["labels"][0].tolist()  # 只看 batch 中第一个样本
        input_ids = inputs["input_ids"][0].tolist()

        # 统计
        total = len(labels)
        ignored = sum(l == -100 for l in labels)
        valid = total - ignored

        print("\n🧪 [Debug] Loss Mask 检查（仅打印一次）")
        print(f"Total tokens: {total}")
        print(f"Ignored tokens (-100): {ignored}")
        print(f"Tokens contributing to loss: {valid}")
        print(f"Masked ratio: {ignored / total:.2%}")

        # 可选：把 prompt / assistant 边界附近打印出来
        print("\n[前 80 个 label]:")
        print(labels[:80])
        print("\n[前 80 个 token]:")
        print(tokenizer.convert_ids_to_tokens(input_ids[:80]))

        self.printed = True



print("📊 开始动态分析数据集长度...")


dataset = MsDataset.load("GDUTSONG/EHRSQL_2024_CoT_Train", split="train")
datasets_to_merge = [dataset]
dataset_for_analysis = concatenate_datasets(datasets_to_merge) if datasets_to_merge else None
print(f"📊 数据集加载完成，共 {len(dataset_for_analysis)} 条样本。")


# 1. 配置模型和分词器
# ==============================================================================
dtype = None  # 自动检测
load_in_4bit = False  #4-bit量化

print("\n🚀 开始加载模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,  # 使用之前定义的变量
    max_seq_length=max_seq_length, # 👈 使用动态计算出的长度
    dtype=dtype,
    #fast_inference = True,
    load_in_4bit=load_in_4bit,
    #gpu_memory_utilization = 0.5
)
print("✅ 模型加载完成！")


# 2. 配置 LoRA
# ==============================================================================
print("🛠️ 开始配置 LoRA 适配器...")
model = FastLanguageModel.get_peft_model(
    model,
    r=Lora_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=Lora_R*2,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print("✅ LoRA 配置完成！")


# 3. 数据准备
# ==============================================================================
EOS_TOKEN = tokenizer.eos_token # "<|im_end|>"
# 定义你的系统提示词
SYS_PROMPT = """
Please concisely summarize your thought process using the format below:
<thinking>
- **Analyze the User's Intent:**...
- **Consider Conversation History:**...
- **Map to Schema:**...
- **Formulate a Plan:**...
</thinking>
```sql
-- Your SQL query
```
"""

import os

# 定义并行处理的核心数 (一个安全且高效的常用设置)
# 假设你有 32 核，这里就会用 30 个来处理数据
NUM_WORKERS = max(1, os.cpu_count() - 2) 
import random

def formatting_prompts_func(examples, mode=mode):
    # 1. 根据 mode 选择对应的 prompt 和 response 列
    if mode == "sql":
        prompts = examples["prompt_d"]
        responses = examples["sql"]
    else:
        prompts = examples["prompt"]
        responses = examples["CoT"]
    
    messages_list = []
    Text='''
\nPlease determine whether the current question can be answered; if you confirm the dataset is unanswerable, refuse to answer and output: 'SELECT 'Unable to answer' AS result;'\n
'''
    for prompt, content in zip(prompts, responses):
        # 2. 如果是 sql 模式，给内容加上 markdown 格式
        if mode == "sql":
            content = f"```sql\n{content}\n```"
            # 原来的操作
            other_str = extract_question_content(prompt)
            prompt = prompt.replace(Text, '')

            #if other_str.strip():  # 检查是否为空或仅包含空白字符
            #    if random.random() < 0.5:
            #        prompt += f"\n\n{other_str}"
    
        messages = [
            #{"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt.replace(Text, '')},
            {"role": "assistant", "content": content},
        ]
        messages_list.append(messages)
    
    texts = [tokenizer.apply_chat_template(messages, tokenize=False) for messages in messages_list]
    
    return {"text": texts}



def filter_by_token_length(example):
    # 不加 special tokens，不截断，真实 token 数
    token_ids = tokenizer(
        example["text"],
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False
    ).input_ids

    return len(token_ids) <= max_seq_length



print("📖 开始加载和处理数据集...")

dataset = (
    dataset_for_analysis
        .map(
            formatting_prompts_func,
            batched=True,
            remove_columns=dataset_for_analysis.column_names,
            # 👇 开启并行处理
            num_proc=NUM_WORKERS 
        )
        .filter(
            filter_by_token_length,
            # 👇 filter 函数同样可以并行化！
            num_proc=NUM_WORKERS
        )
        .shuffle(seed=8192)
)



print("✅ 数据集处理完成！")
print(f"数据示例:\n{dataset[0]['text']}")

print("🚂 开始配置训练器...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length, 
    packing=False,
    assistant_only_loss=True,   # 👈 就加在这里
    callbacks=[DebugLossMaskCallback()],  # 👈 就加这一行
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        warmup_ratio=0.05,
        num_train_epochs=epochs,
        learning_rate=5e-5,
        save_strategy="epoch",
        logging_steps=16,
        optim="adamw_8bit",
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=output_dir,
        report_to="none",
        torch_compile=False
    ),
)
print("✅ 训练器配置完成！")


# 打印GPU信息 (Trainer会自动检测并使用所有可见的GPU)
print(f"检测到 {torch.cuda.device_count()} 个可用的 GPU。")

print("🔥 开始训练！")
trainer_stats = trainer.train()

# 打印训练结束后的统计信息
print(f"训练总用时: {trainer_stats.metrics['train_runtime']:.2f} 秒")


# 5. 推理和保存
# ==============================================================================
print("✨ 训练完成，准备保存模型...")

FastLanguageModel.for_inference(model)

if trainer.is_world_process_zero():
    print(f"\n正在将 LoRA 适配器保存到 {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ LoRA 适配器已成功保存！")

    print(f"合并模型并保存为16位到 {merged_output_dir}...")
    model.save_pretrained_merged(merged_output_dir, tokenizer, save_method="merged_16bit")
    print(f"✅ 16位全量模型已保存！")