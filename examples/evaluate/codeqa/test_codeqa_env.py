#!/usr/bin/env python3
"""
Standalone test for the CodeQA environment.
Run on CPU (no GPU needed):
  cd /projects/p32992/VAGEN
  PYTHONPATH=. python examples/evaluate/codeqa/test_codeqa_env.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from vagen.envs.codeqa.codeqa_env import CodeQA
from vagen.envs.codeqa.utils.answer_extraction import extract_answer_letter


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


def test_answer_extraction():
    """Test the answer extraction utility."""
    print("\n=== Test: Answer Extraction ===")
    cases = [
        ("ANSWER: A", "A"),
        ("ANSWER: Not in context", "NOT_IN_CONTEXT"),
        ("The answer is B", "B"),
        ("My final answer: C", "C"),
        ("I choose D", "D"),
        ("(A)", "A"),
        ("B", "B"),
        ("option C", "C"),
        ("some random text with D in it", "D"),
        ("no answer here at all", "NONE"),
    ]
    passed = 0
    for response, expected in cases:
        result = extract_answer_letter(response)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            print(f"  {status}: '{response}' -> got '{result}', expected '{expected}'")
        else:
            passed += 1
    print(f"  {passed}/{len(cases)} passed")
    return passed == len(cases)


async def test_env_lifecycle():
    """Test the full environment lifecycle."""
    print("\n=== Test: Environment Lifecycle ===")

    # 1. Construction
    env = CodeQA(ENV_CONFIG)
    print(f"  Loaded {len(env.samples)} samples (expected 125 for clean_nt)")
    assert len(env.samples) == 125, f"Expected 125, got {len(env.samples)}"

    # 2. System prompt
    sys_obs = await env.system_prompt()
    assert "obs_str" in sys_obs
    assert "expert code analyst" in sys_obs["obs_str"]
    print(f"  System prompt: {sys_obs['obs_str'][:80]}...")

    # 3. Reset (Turn 1 - OCR)
    obs, info = await env.reset(seed=0)
    assert "obs_str" in obs
    assert "<image>" in obs["obs_str"]
    assert "multi_modal_input" in obs
    images = obs["multi_modal_input"]["<image>"]
    print(f"  Reset seed=0: sample_id={info['sample_id']}, repo={info['repo']}")
    print(f"    num_images={info['num_images']}, tokens={info['vision_tokens_used']}")
    print(f"    obs_str preview: {obs['obs_str'][:60]}...")
    assert len(images) > 0, "Expected at least 1 image"
    assert info["num_images"] == len(images)

    # 4. Step 1 (OCR response -> QA question)
    obs2, reward, done, info2 = await env.step("def hello():\n    pass\n# OCR transcription")
    assert reward == 0.0, f"Turn 1 should have reward=0, got {reward}"
    assert done is False, "Turn 1 should not be done"
    assert "multi_modal_input" not in obs2, "Turn 2 should have no images"
    assert "question" not in obs2.get("obs_str", "").lower() or "answer" in obs2["obs_str"].lower()
    print(f"  Step 1 (OCR): reward={reward}, done={done}")
    print(f"    QA obs preview: {obs2['obs_str'][:100]}...")

    # 5. Step 2 (QA answer - correct)
    correct_letter = env._current_sample["correct_letter"]
    obs3, reward, done, info3 = await env.step(f"ANSWER: {correct_letter}")
    assert done is True, "Turn 2 should be done"
    assert reward == 1.0, f"Correct answer should give reward=1.0, got {reward}"
    assert info3["success"] is True
    assert info3["predicted_letter"] == correct_letter.upper()
    print(f"  Step 2 (QA correct): reward={reward}, done={done}, success={info3['success']}")
    print(f"    predicted={info3['predicted_letter']}, correct={info3['correct_letter']}")

    # 6. Close
    await env.close()
    print("  Close: OK")

    # 7. Test wrong answer
    env2 = CodeQA(ENV_CONFIG)
    await env2.reset(seed=0)
    await env2.step("OCR text here")
    wrong_letter = "A" if correct_letter.upper() != "A" else "B"
    obs4, reward, done, info4 = await env2.step(f"ANSWER: {wrong_letter}")
    assert reward == 0.0, f"Wrong answer should give reward=0.0, got {reward}"
    assert info4["success"] is False
    print(f"  Wrong answer test: reward={reward}, success={info4['success']}")
    await env2.close()

    print("  All lifecycle tests PASSED")
    return True


async def test_seed_mapping():
    """Verify deterministic seed-to-sample mapping."""
    print("\n=== Test: Seed Mapping ===")
    env = CodeQA(ENV_CONFIG)

    # Same seed should give same sample
    obs1, info1 = await env.reset(seed=42)
    sample_id_1 = info1["sample_id"]

    obs2, info2 = await env.reset(seed=42)
    sample_id_2 = info2["sample_id"]

    assert sample_id_1 == sample_id_2, f"Same seed gave different samples: {sample_id_1} vs {sample_id_2}"
    print(f"  seed=42 -> {sample_id_1} (consistent)")

    # Different seeds should give different samples (usually)
    obs3, info3 = await env.reset(seed=43)
    sample_id_3 = info3["sample_id"]
    print(f"  seed=43 -> {sample_id_3}")
    assert sample_id_1 != sample_id_3, "Adjacent seeds should map to different samples"

    # All 125 seeds should cover all samples
    seen_ids = set()
    for s in range(125):
        _, info = await env.reset(seed=s)
        seen_ids.add(info["sample_id"])
    print(f"  Seeds 0-124: {len(seen_ids)} unique samples covered")
    assert len(seen_ids) == 125, f"Expected 125 unique samples, got {len(seen_ids)}"

    await env.close()
    print("  All seed mapping tests PASSED")
    return True


async def test_registry():
    """Test that CodeQA is registered in the environment registry."""
    print("\n=== Test: Registry ===")
    from vagen.envs.registry import get_env_cls, list_envs
    envs = list_envs()
    print(f"  Available envs: {envs}")
    assert "CodeQA" in envs, f"CodeQA not in registry: {envs}"
    cls = get_env_cls("CodeQA")
    assert cls is CodeQA, f"Registry returned wrong class: {cls}"
    print("  Registry test PASSED")
    return True


async def main():
    print("=" * 60)
    print("CodeQA Environment Test Suite")
    print("=" * 60)

    results = {}

    # Test answer extraction (sync)
    results["answer_extraction"] = test_answer_extraction()

    # Test registry
    results["registry"] = await test_registry()

    # Test environment lifecycle
    results["lifecycle"] = await test_env_lifecycle()

    # Test seed mapping
    results["seed_mapping"] = await test_seed_mapping()

    # Summary
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
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
