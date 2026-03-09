IMAGE_PLACEHOLDER = "<image>"


def system_prompt() -> str:
    """System prompt for the CodeQA two-turn environment."""
    return (
        "You are an expert code analyst. You will be given code from a Python "
        "repository as images. Your task has two parts:\n"
        "1. First, you will be shown code images. Read and transcribe the code carefully.\n"
        "2. Then, you will be asked a multiple choice question about the code. "
        "Answer with exactly one letter: A, B, C, or D.\n\n"
        "Base your answers ONLY on the code shown in the images."
    )


def ocr_observation(num_images: int) -> str:
    """
    Build the Turn 1 observation: code images + OCR instruction.
    Returns obs_str with <image> placeholders.
    """
    image_placeholders = "\n".join([IMAGE_PLACEHOLDER] * num_images)
    return (
        f"Here is the source code of a Python repository, rendered as images:\n\n"
        f"{image_placeholders}\n\n"
        f"Please read and transcribe the code shown in these images. "
        f"Provide a faithful text representation of all the code you can see."
    )


def qa_observation(question: str) -> str:
    """
    Build the Turn 2 observation: MCQ question (text only, no images).
    """
    return (
        f"Now answer the following multiple choice question based on the code you just read.\n\n"
        f"{question}\n\n"
        f"You MUST select exactly one answer: A, B, C, or D.\n"
        f"State your answer in the format: ANSWER: X"
    )
