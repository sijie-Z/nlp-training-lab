"""
LoRA 模型封装

作用：
1. 加载 Qwen2.5-0.5B 基座模型（4-bit 量化节省显存）
2. 应用 LoRA 配置（在 attention 层插入低秩矩阵）
3. 打印可训练参数量，对比总参数量

用法：
    model, tokenizer = build_lora_model("Qwen/Qwen2.5-0.5B", r=8)
    model.print_trainable_parameters()
    # trainable params: 2,097,152 || all params: 494,032,768 || trainable%: 0.42
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


def build_lora_model(
    model_name="Qwen/Qwen2.5-0.5B",
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=None,
    use_4bit=True,
    device="auto",
):
    """
    构建 LoRA 模型

    参数：
        model_name: 基座模型名称
        r: LoRA 秩（越大→可训练参数越多）
        lora_alpha: 缩放系数（越大→LoRA 影响越大）
        lora_dropout: Dropout 概率
        target_modules: 作用的目标模块（默认 q_proj, v_proj）
        use_4bit: 是否使用 4-bit 量化（RTX 3050 4GB 建议开启）
        device: 设备

    返回：
        (model, tokenizer)
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    # OmegaConf 的列表转 Python 原生列表（JSON 序列化需要）
    target_modules = list(target_modules)

    # 量化配置（4-bit NormalFloat，节省显存）
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # 加载基座模型
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
    )

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA 配置
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 应用 LoRA
    model = get_peft_model(model, lora_config)

    # 打印参数
    print("\n" + "=" * 50)
    model.print_trainable_parameters()
    print("=" * 50 + "\n")

    return model, tokenizer
