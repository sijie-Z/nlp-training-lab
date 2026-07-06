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
import json

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None

# LoRA adapter 路径（项目根目录）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "outputs/checkpoints/lora_adapter")

# 强制 HuggingFace 缓存到 D 盘（如果存在）
if os.path.exists("D:/"):
    os.environ.setdefault("HF_HOME", "D:/packge/huggingface")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "D:/packge/huggingface/hub")


class LLMWorker:
    """LoRA 模型推理器（自动适配 GPU/CPU）"""

    def __init__(self, adapter_path=None, force_cpu=False, demo_mode=None):
        self.adapter_path = adapter_path or ADAPTER_PATH
        self.model = None
        self.tokenizer = None
        self.demo_answers = self._load_demo_answers()
        deps_ready = torch is not None and AutoModelForCausalLM is not None and AutoTokenizer is not None
        self.demo_mode = (not deps_ready) if demo_mode is None else demo_mode

        if self.demo_mode:
            reason = "torch/transformers 未安装" if not deps_ready else "显式启用"
            print(f"[LLM] Demo 模式 ({reason})：不加载大模型，使用本地知识与模板回答")
            return

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
        if self.demo_mode:
            return self._generate_demo(instruction)

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
        if self.demo_mode:
            return self._generate_demo_from_prompt(prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _load_demo_answers(self):
        answers_path = os.path.join(
            PROJECT_ROOT, "experiments/exp004_lora_prep/answers_after_lora.json"
        )
        if not os.path.exists(answers_path):
            return {}
        try:
            with open(answers_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("answers", {})
        except Exception:
            return {}

    def _generate_demo(self, instruction):
        normalized = instruction.strip()
        for question, answer in self.demo_answers.items():
            if question in normalized or normalized in question:
                return answer

        gis_terms = {
            "gis": "GIS（地理信息系统）用于采集、管理、分析和可视化地理空间数据，核心价值是把位置、属性和空间关系结合起来支持决策。",
            "地理信息系统": "地理信息系统是 GIS 的中文名称，它把地图、空间数据、属性数据和空间分析方法组织在一起。",
            "遥感": "遥感是通过卫星、航空器或无人机等平台，在不直接接触目标的情况下获取地表信息的技术。",
            "ndvi": "NDVI 是归一化植被指数，常用公式为 (NIR - Red) / (NIR + Red)，用于反映植被覆盖和生长状况。",
            "坐标": "坐标系统用于描述地理对象的位置。GIS 中常见地理坐标系统、投影坐标系统和不同的大地基准面。",
        }
        lowered = normalized.lower()
        for keyword, answer in gis_terms.items():
            if keyword in lowered or keyword in normalized:
                return answer

        if any(word in normalized for word in ["你可以做什么", "你能做什么", "有什么用", "功能"]):
            return (
                "我现在运行在本地 demo 模式，主要能做三件事："
                "1. 回答常见 GIS 概念问题；"
                "2. 对 QGIS、NDVI、坐标系统等操作类问题走知识库检索；"
                "3. 展示 Router、RAG、LLMWorker、API 和 harness 这条大模型应用链路。"
                "我不是当前机器上的通用大模型聊天本体。真实 Qwen/LoRA 模型需要在有模型文件和依赖的环境中加载。"
            )

        if any(word in normalized for word in ["没有大模型", "没大模型", "不能回答", "模型本体", "真实模型"]):
            return (
                "是的，当前这个页面没有加载真实 Qwen/LoRA 大模型，而是在 demo fallback 模式下运行。"
                "原因是这台电脑没有独立显卡，也没有完整的深度学习依赖和模型本体。"
                "所以它能演示完整产品链路，但不能像真正的大模型一样自由泛聊。"
                "如果模型本体在另一台电脑，可以把模型/adapter 和依赖迁移过来，"
                "或者让另一台电脑启动模型服务，这台电脑只调用远程接口。"
            )

        if any(word in normalized for word in ["你好", "您好", "hello"]):
            return "你好，我是 GeoAI Assistant 的本地演示模式，可以回答 GIS 概念和操作类问题。"

        return (
            "当前处于本地 demo 模式：没有加载 Qwen/LoRA 大模型，但对话链路已经打通。"
            "如果问题属于 GIS 操作类，会优先走知识库检索；换到有 GPU 或完整依赖的环境后，同一接口可切换到 LoRA 推理。"
        )

    def _generate_demo_from_prompt(self, prompt):
        question = self._extract_between(prompt, "问题：", "\n\n### Response:")
        context = self._extract_between(prompt, "参考资料：\n", "\n\n问题：")
        if not context:
            return self._generate_demo(question or prompt)

        snippets = []
        for block in context.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            snippets.append(block)
            if len(snippets) >= 2:
                break

        answer = "根据知识库检索结果："
        if question:
            answer += f"\n问题：{question.strip()}"
        for idx, snippet in enumerate(snippets, start=1):
            answer += f"\n{idx}. {snippet}"
        return answer

    @staticmethod
    def _extract_between(text, start, end):
        if start not in text:
            return ""
        value = text.split(start, 1)[1]
        if end in value:
            value = value.split(end, 1)[0]
        return value.strip()


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "什么是遥感"
    print(f"\n[LLM Test] Query: {query}\n")
    llm = LLMWorker()
    answer = llm.generate(query)
    print(f"\nAnswer:\n{answer}")
