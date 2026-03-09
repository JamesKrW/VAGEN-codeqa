import re


def extract_answer_letter(response: str) -> str:
    """
    Extract the answer letter (A/B/C/D) or "NOT_IN_CONTEXT" from model response.

    Handles various formats including:
    - "ANSWER: Not in context" (new format for missing evidence)
    - "ANSWER: A" (explicit format)
    - "The answer is A"
    - "The correct answer is B"
    - "My final answer: C"
    - "I choose D"
    - Standalone letter at end
    - Letter in parentheses: "(A)"

    Ported from AgentOCR scripts_longcodebench_qa/5_run_vlm.py.
    """
    # Priority 0: Check for "Not in context" response
    not_in_context_match = re.search(r'ANSWER:\s*Not\s+in\s+context', response, re.IGNORECASE)
    if not_in_context_match:
        return "NOT_IN_CONTEXT"

    # Also check for variations
    if re.search(r'not\s+(?:found\s+)?in\s+(?:the\s+)?context', response, re.IGNORECASE):
        return "NOT_IN_CONTEXT"
    if re.search(r'cannot\s+(?:find|determine|answer)', response, re.IGNORECASE) and \
       re.search(r'(?:from|in)\s+(?:the\s+)?(?:provided\s+)?(?:code|images)', response, re.IGNORECASE):
        return "NOT_IN_CONTEXT"

    # Priority 1: Explicit ANSWER: pattern (our instructed format)
    answer_match = re.search(r'ANSWER:\s*([A-D])', response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    # Priority 2: "final answer" patterns (strong indicator)
    final_match = re.search(r'final\s+answer[:\s]+(?:is\s+)?([A-D])', response, re.IGNORECASE)
    if final_match:
        return final_match.group(1).upper()

    # Priority 3: "the answer is X" / "correct answer is X" patterns
    answer_is_match = re.search(r'(?:the\s+)?(?:correct\s+)?answer\s+(?:is|would\s+be|should\s+be)[:\s]+([A-D])', response, re.IGNORECASE)
    if answer_is_match:
        return answer_is_match.group(1).upper()

    # Priority 4: "I choose/select X" patterns
    choose_match = re.search(r'(?:I\s+)?(?:choose|select|pick)\s+([A-D])', response, re.IGNORECASE)
    if choose_match:
        return choose_match.group(1).upper()

    # Priority 5: Letter in parentheses at end like "(A)" or "( A )"
    paren_match = re.search(r'\(\s*([A-D])\s*\)\s*\.?\s*$', response.strip(), re.IGNORECASE)
    if paren_match:
        return paren_match.group(1).upper()

    # Priority 6: Standalone letter at the end
    end_match = re.search(r'\b([A-D])\s*\.?\s*$', response.strip(), re.IGNORECASE)
    if end_match:
        return end_match.group(1).upper()

    # Priority 7: "option X" patterns
    option_match = re.search(r'option\s+([A-D])', response, re.IGNORECASE)
    if option_match:
        return option_match.group(1).upper()

    # Priority 8: Fallback - first letter A-D found (risky, but better than NONE)
    first_match = re.search(r'\b([A-D])\b', response)
    if first_match:
        return first_match.group(1).upper()

    return "NONE"
