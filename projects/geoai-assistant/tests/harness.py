"""
GeoAI Assistant acceptance harness.

This script runs a small, repeatable question set against ChatPipeline and checks
whether the end-to-end behavior still meets the demo acceptance bar.

Usage:
    python projects/geoai-assistant/tests/harness.py
    python projects/geoai-assistant/tests/harness.py --json-output outputs/geoai_harness.json
    python projects/geoai-assistant/tests/harness.py --real-model --cpu
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(PROJECT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)

from chat_pipeline import ChatPipeline


@dataclass
class HarnessCase:
    name: str
    query: str
    expected_source: Optional[str] = None
    required_terms: Optional[List[str]] = None
    min_references: int = 0
    max_time_ms: Optional[int] = None


@dataclass
class HarnessResult:
    name: str
    query: str
    passed: bool
    source: str
    time_ms: int
    answer_preview: str
    failures: List[str]
    references: List[dict]


CASES = [
    HarnessCase(
        name="gis_term_remote_sensing",
        query="什么是遥感",
        expected_source="lora",
        required_terms=["遥感", "卫星"],
        max_time_ms=3000,
    ),
    HarnessCase(
        name="rag_qgis_import_shp",
        query="QGIS怎么导入shp文件",
        expected_source="rag",
        required_terms=["QGIS", "导入"],
        min_references=1,
        max_time_ms=3000,
    ),
    HarnessCase(
        name="lora_trained_identity",
        query="你是谁",
        expected_source="lora",
        required_terms=["问答助手"],
        max_time_ms=3000,
    ),
    HarnessCase(
        name="gis_term_ndvi",
        query="NDVI怎么计算",
        expected_source="rag",
        required_terms=["NDVI"],
        min_references=1,
        max_time_ms=3000,
    ),
    HarnessCase(
        name="general_greeting",
        query="你好",
        expected_source="lora",
        required_terms=["GeoAI Assistant"],
        max_time_ms=3000,
    ),
]


def evaluate_case(case, pipeline):
    t0 = time.time()
    result = pipeline.chat(case.query)
    wall_time_ms = int((time.time() - t0) * 1000)
    source = result.get("source", "")
    answer = result.get("answer", "")
    references = result.get("references") or []
    reported_time_ms = result.get("time_ms", wall_time_ms)

    failures = []
    if case.expected_source and source != case.expected_source:
        failures.append(f"expected source={case.expected_source}, got {source}")

    for term in case.required_terms or []:
        haystack = answer + " " + " ".join(ref.get("title", "") for ref in references)
        if term not in haystack:
            failures.append(f"missing required term: {term}")

    if len(references) < case.min_references:
        failures.append(
            f"expected at least {case.min_references} references, got {len(references)}"
        )

    if case.max_time_ms is not None and wall_time_ms > case.max_time_ms:
        failures.append(f"expected <= {case.max_time_ms}ms, got {wall_time_ms}ms")

    return HarnessResult(
        name=case.name,
        query=case.query,
        passed=not failures,
        source=source,
        time_ms=reported_time_ms,
        answer_preview=answer.replace("\n", " ")[:120],
        failures=failures,
        references=references,
    )


def print_report(results):
    print("\nGeoAI Assistant Harness")
    print("=" * 80)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}")
        print(f"  query:  {result.query}")
        print(f"  source: {result.source}  time_ms={result.time_ms}")
        print(f"  answer: {result.answer_preview}")
        if result.references:
            titles = ", ".join(ref.get("title", "") for ref in result.references[:3])
            print(f"  refs:   {titles}")
        for failure in result.failures:
            print(f"  - {failure}")
    print("=" * 80)
    passed = sum(1 for result in results if result.passed)
    print(f"Summary: {passed}/{len(results)} passed")


def write_json(path, results):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "passed": all(result.passed for result in results),
        "total": len(results),
        "passed_count": sum(1 for result in results if result.passed),
        "results": [asdict(result) for result in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON report written to: {path}")


def main():
    parser = argparse.ArgumentParser(description="GeoAI Assistant acceptance harness")
    parser.add_argument("--cpu", action="store_true", help="force CPU in real-model mode")
    parser.add_argument(
        "--real-model",
        action="store_true",
        help="try Qwen/LoRA instead of the default demo fallback",
    )
    parser.add_argument("--json-output", help="optional path for a JSON report")
    args = parser.parse_args()

    pipeline = ChatPipeline(force_cpu=args.cpu, demo_mode=not args.real_model)
    results = [evaluate_case(case, pipeline) for case in CASES]

    print_report(results)
    if args.json_output:
        write_json(args.json_output, results)

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
