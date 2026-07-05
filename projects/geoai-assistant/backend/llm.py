"""
LoRA 模型推理模块（CPU 兼容版）

支持两种模式：
1. GPU 模式：4bit 量化加载（显存 ~1GB，速度快）
2. CPU 模式：float32 加载（内存 ~3GB，速度较慢但核显也能跑）

自动检测 CUDA 可用性选择模式。

用法：
    llm = LLMWorker()
    answer = llm.generate("什么是遥感")
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# LoRA adapter 路径（项目根目录）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "outputs/checkpoints/lora_adapter")

# 强制 HuggingFace 缓存到 D 盘（如果存在）
if os.path.exists("D:/"):
    os.environ.setdefault("HF_HOME", "D:/packge/huggingface")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "D:/packge/huggingface/hub")


class LLMWorker:
    """LoRA 模型推理器（自动适配 GPU/CPU）"""

    def __init__(self, adapter_path=None, force_cpu=False):
        self.adapter_path = adapter_path or ADAPTER_PATH
        self.use_gpu = torch.cuda.is_available() and not force_cpu

        print(f"[LLM] 设备: {'CUDA (4bit)' if self.use_gpu else 'CPU (float32)'}")

        if self.use_gpu:
            self._load_gpu()
        else:
            self._load_cpu()

    def _load_gpu(self):
        """GPU 模式：4bit 量化"""
        print("[LLM] 加载基座模型 (Qwen2.5-0.5B, 4bit)...")
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

        # 尝试加载 LoRA adapter
        if os.path.exists(self.adapter_path):
            from peft import PeftModel
            print("[LLM] 加载 LoRA adapter...")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            print(f"[LLM] LoRA adapter 已加载")
        else:
            print(f"[LLM] 未找到 adapter ({self.adapter_path})，使用基座模型")

        self.model.eval()
        print(f"[LLM] 显存: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    def _load_cpu(self):
        """CPU 模式：float32 全精度"""
        print("[LLM] 加载基座模型 (Qwen2.5-0.5B, float32 CPU)...")
        print("[LLM] 提示：首次加载需下载模型（~1GB），之后使用缓存")

        # CPU 上用 float32 加载，内存约 2GB
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 尝试加载 LoRA adapter
        if os.path.exists(self.adapter_path):
            from peft import PeftModel
            print("[LLM] 加载 LoRA adapter (CPU)...")
            self.model = PeftModel.from_pretrained(
                self.model, self.adapter_path,
                device_map="cpu",
            )
            print(f"[LLM] LoRA adapter 已加载")
        else:
            print(f"[LLM] 未找到 adapter ({self.adapter_path})，使用基座模型")

        self.model.eval()
        print("[LLM] CPU 模型就绪（推理速度较慢，预期 30-120 秒/次）")

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

        # 提取 Response 部分
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
    print(f"\n[LLM Test] Query: {query}\n")
    llm = LLMWorker()
    answer = llm.generate(query)
    print(f"\nAnswer:\n{answer}")
