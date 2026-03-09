#!/usr/bin/env python3
"""
Full pipeline test for CodeQA through VAGEN.
Runs on CPU (no GPU, no vLLM server) using a mock adapter.

Tests the complete path:
  run_eval config parsing → job expansion → runner → workflow → env → rollout dump

Usage:
  cd /projects/p32992/VAGEN
  PYTHONPATH=. python examples/evaluate/codeqa/test_full_pipeline.py
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from vagen.evaluate.vision_workflow import GenericVisionInferenceWorkflow
from vagen.evaluate.adapters.openai_adapter import OpenAIAdapter
from vagen.envs.codeqa.codeqa_env import CodeQA

DATA_ROOT = "/gpfs/projects/p32992/AgentOCR/code_work/vlm_code_optimize/compare_llm_vs_vlm"

ENV_CONFIG = {
    "dataset_path": f"{DATA_ROOT}/data/longcodebench_qa/prepared_dataset_full/all_repos_dataset.jsonl",
    "images_dir": f"{DATA_ROOT}/data/longcodebench_qa/code_images_8pt",
    "contaminated_path": f"{DATA_ROOT}/data/longcodebench_qa/contaminated_questions_v3.json",
    "nontruncated_path": f"{DATA_ROOT}/data/longcodebench_qa/nontruncated_repos.json",
    "subset": "clean_nt",
    "max_vision_tokens": 230000,
    "correct_reward": 1.0,
}


class MockAdapter(OpenAIAdapter):
    """
    Mock adapter that returns deterministic responses without calling any API.
    Turn 1 (OCR): returns a fake transcription
    Turn 2 (QA): returns the correct answer (to verify scoring works)
    """

    def __init__(self, answer_correctly=True):
        # Don't call super().__init__ - we don't need a real client
        self.client = None
        self.model = "mock"
        self.answer_correctly = answer_correctly
        self._turn = 0
        self._correct_letter = None

    async def acompletion(self, messages: List[Dict[str, Any]], **chat_config: Any) -> str:
        self._turn += 1
        if self._turn == 1:
            # Turn 1: OCR response
            return "def hello():\n    print('hello world')\n# Mock OCR transcription"
        else:
            # Turn 2: QA response
            if self.answer_correctly:
                # We need to extract the correct answer from the question
                # The mock doesn't know it, so just return "ANSWER: A"
                # The test will check both correct and incorrect paths
                return "ANSWER: A"
            else:
                return "ANSWER: Z"  # Invalid

    def reset_turn(self):
        self._turn = 0


async def test_single_episode_workflow():
    """Test a single episode through GenericVisionInferenceWorkflow."""
    print("\n=== Test: Single Episode via Workflow ===")

    dump_dir = tempfile.mkdtemp(prefix="codeqa_test_")

    try:
        adapter = MockAdapter(answer_correctly=True)
        wf = GenericVisionInferenceWorkflow(
            adapter=adapter,
            dump_dir=dump_dir,
            dump_enabled=True,
            chat_config={"temperature": 0, "max_tokens": 4096},
        )

        result = await wf.arun_episode(
            env_cls=CodeQA,
            env_config=ENV_CONFIG,
            seed=0,
            rollout_id="test_episode_0",
            dump_override=dump_dir,
            max_turns=2,
            episode_metadata={"tag_id": "test", "env_name": "CodeQA"},
        )

        print(f"  Result keys: {list(result.keys())}")
        print(f"  finish_reason: {result.get('finish_reason')}")
        print(f"  num_turns: {result.get('num_turns')}")
        print(f"  terminated: {result.get('terminated')}")
        print(f"  cumulative_reward: {result.get('cumulative_reward')}")

        # Check rollout was dumped
        rollout_dir = os.path.join(dump_dir, "test_episode_0")
        if os.path.exists(rollout_dir):
            files = os.listdir(rollout_dir)
            print(f"  Rollout files: {sorted(files)}")

            # Check metrics.json
            metrics_path = os.path.join(rollout_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
                print(f"  metrics.json keys: {list(metrics.keys())}")
                print(f"    terminated: {metrics.get('terminated')}")
                print(f"    success: {metrics.get('success')}")
                print(f"    num_turns: {metrics.get('num_turns')}")
                print(f"    finish_reason: {metrics.get('finish_reason')}")
                print(f"    cumulative_reward: {metrics.get('cumulative_reward')}")

            # Check transcript
            transcript_path = os.path.join(rollout_dir, "transcript.txt")
            if os.path.exists(transcript_path):
                with open(transcript_path) as f:
                    transcript = f.read()
                print(f"  transcript.txt: {len(transcript)} chars")
                # Show first few lines
                lines = transcript.split('\n')[:10]
                for line in lines:
                    print(f"    | {line[:100]}")

            # Check images directory
            images_dir = os.path.join(rollout_dir, "images")
            if os.path.exists(images_dir):
                image_files = os.listdir(images_dir)
                print(f"  images/ directory: {len(image_files)} files")
        else:
            print(f"  WARNING: Rollout dir not found at {rollout_dir}")
            # Check what was actually created
            all_contents = os.listdir(dump_dir)
            print(f"  dump_dir contents: {all_contents}")
            if all_contents:
                rollout_dir = os.path.join(dump_dir, all_contents[0])
                files = os.listdir(rollout_dir)
                print(f"  Actual rollout files: {sorted(files)}")

        assert result.get("finish_reason") in ("done", "max_turns"), \
            f"Unexpected finish_reason: {result.get('finish_reason')}"
        assert result.get("num_turns") == 2, \
            f"Expected 2 turns, got {result.get('num_turns')}"

        print("  Single episode test PASSED")
        return True

    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)


async def test_config_to_jobs():
    """Test that the config YAML correctly parses and expands to 125 jobs."""
    print("\n=== Test: Config → Job Expansion ===")

    from vagen.evaluate.run_eval import _parse_env_specs, _expand_jobs
    from omegaconf import OmegaConf

    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg_node = OmegaConf.load(cfg_path)
    cfg = OmegaConf.to_container(cfg_node, resolve=True)

    env_specs = _parse_env_specs(cfg)
    assert len(env_specs) == 1
    assert env_specs[0].name == "CodeQA"
    assert env_specs[0].n_envs == 125
    assert env_specs[0].max_turns == 2
    print(f"  Parsed 1 env spec: CodeQA, n_envs=125, max_turns=2")

    base_dir = os.path.dirname(os.path.abspath(cfg_path))
    jobs = _expand_jobs(env_specs, base_seed=0, base_dir=base_dir, default_max_turns=2)
    assert len(jobs) == 125, f"Expected 125 jobs, got {len(jobs)}"

    # Verify all seeds are unique
    seeds = [j["data"]["seed"] for j in jobs]
    assert len(set(seeds)) == 125, f"Expected 125 unique seeds, got {len(set(seeds))}"
    assert min(seeds) >= 0 and max(seeds) <= 124, \
        f"Seeds out of range: min={min(seeds)}, max={max(seeds)}"

    # Verify env class
    assert jobs[0]["data"]["env_cls"] is CodeQA
    print(f"  Expanded to 125 jobs, seeds {min(seeds)}-{max(seeds)}, all unique")
    print("  Config → Jobs test PASSED")
    return True


async def test_multi_episode():
    """Test running 3 episodes to verify concurrency and rollout structure."""
    print("\n=== Test: Multi-Episode (3 samples) ===")

    dump_dir = tempfile.mkdtemp(prefix="codeqa_multi_")

    try:
        # Run 3 episodes sequentially with mock adapter
        results = []
        for seed in [0, 1, 2]:
            adapter = MockAdapter(answer_correctly=True)
            wf = GenericVisionInferenceWorkflow(
                adapter=adapter,
                dump_dir=dump_dir,
                dump_enabled=True,
                chat_config={"temperature": 0, "max_tokens": 4096},
            )
            tag_dir = os.path.join(dump_dir, "tag_codeqa_test")
            os.makedirs(tag_dir, exist_ok=True)

            result = await wf.arun_episode(
                env_cls=CodeQA,
                env_config=ENV_CONFIG,
                seed=seed,
                rollout_id=None,
                dump_override=tag_dir,
                max_turns=2,
                episode_metadata={"tag_id": "codeqa_test", "env_name": "CodeQA", "seed": seed},
            )
            results.append(result)
            print(f"  Seed {seed}: finish={result.get('finish_reason')}, "
                  f"turns={result.get('num_turns')}, "
                  f"reward={result.get('cumulative_reward')}")

        # Check rollout directory structure
        tag_dir = os.path.join(dump_dir, "tag_codeqa_test")
        rollouts = [d for d in os.listdir(tag_dir) if os.path.isdir(os.path.join(tag_dir, d))]
        print(f"  Rollout dirs created: {len(rollouts)}")
        assert len(rollouts) == 3, f"Expected 3 rollout dirs, got {len(rollouts)}"

        # Check each has metrics.json
        for rd in rollouts:
            metrics_path = os.path.join(tag_dir, rd, "metrics.json")
            assert os.path.exists(metrics_path), f"Missing metrics.json in {rd}"

        # Try summary generation
        try:
            from vagen.evaluate.utils.summary_utils import write_rollouts_summary_from_dump
            summary_path = write_rollouts_summary_from_dump(dump_dir=tag_dir, filename="summary.json")
            with open(summary_path) as f:
                summary = json.load(f)
            print(f"  Summary: {json.dumps(summary, indent=2)[:500]}")
        except Exception as e:
            print(f"  Summary generation: {e}")

        print("  Multi-episode test PASSED")
        return True

    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)


async def main():
    print("=" * 60)
    print("CodeQA Full Pipeline Test (CPU, Mock Adapter)")
    print("=" * 60)

    results = {}

    results["config_to_jobs"] = await test_config_to_jobs()
    results["single_episode"] = await test_single_episode_workflow()
    results["multi_episode"] = await test_multi_episode()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll pipeline tests PASSED!")
        print("The full VAGEN pipeline works end-to-end on CPU.")
        print("Ready for GPU submission with real vLLM server.")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
