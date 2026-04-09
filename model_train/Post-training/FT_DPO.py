# -*- coding: utf-8 -*-
"""
### Unsloth - Qwen2 DPO 全量数据训练脚本
"""
from unsloth import PatchDPOTrainer, FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
import pprint

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# 定义路径变量，方便管理
model_name = os.path.join(BASE_DIR, "MODEL", "Qwne3_4B")
# Lora
output_dir = os.path.join(SCRIPT_DIR, "model", "4B_Pro_DPO")
# 2. 数据准备 (Data Prep)
jsonl_file_path = "EHRSQL_DPO_Dataset.jsonl"#还没有提供

# 首先对DPOTrainer进行补丁
PatchDPOTrainer()

# 1. 模型加载配置
# 将最大序列长度设置为8192
max_seq_length = 8192
dtype = None  # 自动检测, 对于Ampere架构及以上推荐 bfloat16
load_in_4bit = False  # 使用4-bit量化以节省显存

print("正在加载Qwen2模型...")
# 加载Qwen2模型
# Unsloth提供了针对Qwen2优化过的版本，加载速度更快，训练更高效
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,  
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "YOUR_HF_TOKEN", # 如果需要访问gated model，请填入Hugging Face token
)
print("模型加载完成。")



print("正在加载全量数据进行训练...")
# 直接加载整个文件作为训练集，无需切分
# `split='train'` 是datasets库的默认用法，当只有一个文件时它会加载所有数据
train_dataset = load_dataset("json", data_files=jsonl_file_path, split="train")

print("数据加载完成！")
print(f"数据集信息: {train_dataset}")
print(f"数据总条数: {len(train_dataset)}")
# JSONL中的无关字段 (如 id, type) 会被加载，但DPOTrainer会忽略它们，所以没关系。
print(f"数据集包含的字段: {train_dataset.column_names}")

# 打印一条数据样本，检查格式是否正确
print("\n--- 数据样本预览 ---")
row = train_dataset[0]
print("Prompt:")
pprint.pprint(row["prompt"])
print("\nChosen:")
pprint.pprint(row["chosen"])
print("\nRejected:")
pprint.pprint(row["rejected"])
print("---------------------\n")


# 3. 添加LoRA适配器
print("正在配置LoRA适配器...")
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # LoRA秩
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth", # 节省显存的关键
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
print("LoRA配置完成。")

# 4. 训练DPO模型
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # 设置为None，TRL会自动创建参考模型
    args=DPOConfig(
        per_device_train_batch_size=1,  # 由于序列长度增加，减小批次大小以防止OOM
        gradient_accumulation_steps=8,  # 增加梯度累积步数，维持有效批次大小 (1*8=8)
        warmup_ratio=0.1,
        num_train_epochs=1,
        learning_rate=5e-6,
        logging_steps=20,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=output_dir, # 新的输出目录
        report_to="none",
    ),
    beta=0.1,
    train_dataset=train_dataset, # <<< 使用全量数据集
    eval_dataset=None,          # <<< 不进行验证
    tokenizer=tokenizer,
    max_length=max_seq_length,
    max_prompt_length=max_seq_length - 1024, # 提示最大长度 (7168)
)

print("开始DPO训练...")
dpo_trainer.train()
