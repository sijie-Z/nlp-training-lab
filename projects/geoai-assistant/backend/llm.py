"""
LoRA 模型推理模块

加载 Qwen2.5-0.5B + LoRA adapter，回答 GIS 相关问题。

用法：
    llm = LLMWorker()
    answer = llm.generate("什么是遥感")
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# LoRA adapter 路径（项目根目录：nlp-training-lab/）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "outputs/checkpoints/lora_adapter")


class LLMWorker:
    """LoRA 模型推理器"""

    def __init__(self, adapter_path=None):
        self.adapter_path = adapter_path or ADAPTER_PATH

        print(f"[LLM] 加载基座模型...")
        # 4bit 量化加载 Qwen
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载 LoRA adapter
        if os.path.exists(self.adapter_path):
            print(f"[LLM] 加载 LoRA adapter...")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            print(f"[LLM] LoRA adapter 已加载")
        else:
            print(f"[LLM] 未找到 LoRA adapter（{self.adapter_path}），使用基座模型")

        self.model.eval()
        if torch.cuda.is_available():
            print(f"[LLM] 显存占用: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    def generate(self, instruction, max_new_tokens=100):
        """给定指令，生成回答"""
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Response:\n" in full:
            return full.split("### Response:\n")[-1].strip()
        return full

    def generate_with_prompt(self, prompt, max_new_tokens=100):
        """直接给定完整 prompt 生成"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "什么是遥感"
    print(f"[LLM Test] Query: {query}\n")
    llm = LLMWorker()
    answer = llm.generate(query)
    print(f"Answer:\n{answer}")
