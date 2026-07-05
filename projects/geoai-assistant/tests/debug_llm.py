"""LoRA 模型调试"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__) + "/../backend")

from llm import LLMWorker

llm = LLMWorker()

# 测试训练数据中的问题
test_qs = ["你是谁", "一加一等于几", "苏州科技大学在哪"]
for q in test_qs:
    prompt = f"### Instruction:\n{q}\n\n### Response:\n"
    inputs = llm.tokenizer(prompt, return_tensors="pt").to(llm.model.device)

    with torch.no_grad():
        outputs = llm.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=llm.tokenizer.eos_token_id,
        )

    full = llm.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nQ: {q}")
    print(f"Full output: {repr(full[:200])}")
    if "### Response:\n" in full:
        answer = full.split("### Response:\n")[-1].strip()
        print(f"Parsed: {answer}")
