# 说明：
# 本文件主要用于训练和微调 Qwen 模型。
# 功能模块：
# 1. 数据集加载：加载训练数据和验证数据，并转换为模型可用的格式。
# 2. 模型加载与微调：加载预训练模型，应用 LoRA 配置进行微调。
# 3. 模型保存：将微调后的模型和分词器保存到指定路径。
# 4. 支持混合精度训练和显存优化。
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
import torch

class QwenDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"问题: {item['input']}\n答案:"
        output_text = item['output']
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            output_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs["labels"] = labels["input_ids"]
        return {key: val.squeeze(0) for key, val in inputs.items()}

def load_training_data(file_path):
    """
    加载训练数据并转换为模型可用的格式
    :param file_path: str, JSONL 文件路径
    :return: list, 包含训练数据的字典列表
    """
    training_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            question = data.get("question", "")
            answer = data.get("answer", "")
            if question and answer:
                training_data.append({"input": question, "output": answer})
    return training_data

# 加载验证数据
def load_validation_data(file_path):
    """
    加载验证数据并转换为模型可用的格式
    :param file_path: str, JSONL 文件路径
    :return: list, 包含验证数据的字典列表
    """
    validation_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            question = data.get("question", "")
            answer = data.get("answer", "")
            if question and answer:
                validation_data.append({"input": question, "output": answer})
    return validation_data

if __name__ == "__main__":
    # -----------------------
    # 初始化文档引擎
    # -----------------------
    model_path = "Models/qwen_model_lora"
    tokenizer_path = "Models/qwen_tokenizer_lora"
    test_path = "data/test.jsonl"
    """"""
    # 加载训练数据
    train_file_path = "data/train.jsonl"
    training_data = load_training_data(train_file_path)
    print(f"加载了 {len(training_data)} 条训练数据")

    # 加载验证集
    val_file_path = "data/val.jsonl"
    validation_data = load_validation_data(val_file_path)
    print(f"加载了 {len(validation_data)} 条验证数据")

        
    # 加载千问模型和分词器
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 使用 4-bit 量化加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        load_in_4bit=True,  # 启用 4-bit 量化
        device_map="auto"   # 自动分配设备
    )

    # 确保模型在 GPU 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 配置 LoRA
    lora_config = LoraConfig(
        r=8,  # LoRA 的秩
        lora_alpha=32,  # LoRA 的缩放因子
        target_modules=["q_proj", "v_proj"],  # 指定需要应用 LoRA 的模块
        lora_dropout=0.1,  # LoRA 的 dropout 概率
        bias="none",  # 不训练偏置
        task_type="CAUSAL_LM"  # 任务类型：因果语言建模
    )

    # 将 LoRA 配置应用到模型
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数数量

    # 创建验证集数据集和数据加载器
    train_dataset = QwenDataset(training_data, tokenizer)
    val_dataset = QwenDataset(validation_data, tokenizer)
    #torch.cuda.empty_cache()

    training_args = TrainingArguments(
        output_dir="Models/qwen_model_lora",
        evaluation_strategy="epoch",  # 每个 epoch 评估一次
        learning_rate=5e-4,  # 微调时使用较大的学习率
        per_device_train_batch_size=1,  # 小批量大小
        gradient_accumulation_steps=4,  # 梯度累积
        num_train_epochs=0.2,  # 训练轮数
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        fp16=True,  # 启用混合精度训练
        optim="paged_adamw_8bit",  # 使用 8-bit Adam 优化器
        report_to="none"  # 禁用报告到外部工具（如 WandB）
    )

    # 在训练前释放显存
    torch.cuda.empty_cache()

    # 更新 Trainer 的构造函数
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # 开始训练
    trainer.train()

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)
    print("微调后的模型已保存")