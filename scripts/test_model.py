"""
Interactive model tester.

Loads a nanochat checkpoint by:
1) source + model tag + step, or
2) explicit checkpoint directory/file path.

Examples:
    # Latest SFT checkpoint by tag
    python -m scripts.test_model --source sft --model-tag parallel-24

    # Exact checkpoint step
    python -m scripts.test_model --source sft --model-tag parallel-24 --step 1200

    # Direct checkpoint file path
    python -m scripts.test_model --checkpoint /vol/nanochat_cache/chatsft_checkpoints/parallel-24/model_001200.pt

    # One-shot prompt mode
    python -m scripts.test_model --source sft --model-tag parallel-24 --prompt "What is 19*23?"

    # Batch mode from file (one question per line)
    python -m scripts.test_model --source sft --model-tag parallel-24 --questions-file questions.txt
"""

import argparse
import os
import re
from contextlib import nullcontext

import torch

from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.engine import Engine
from nanochat.checkpoint_manager import (
    build_model,
    find_last_step,
    load_model,
)


def _parse_step_from_model_file(model_path: str) -> int:
    name = os.path.basename(model_path)
    match = re.fullmatch(r"model_(\d+)\.pt", name)
    if not match:
        raise ValueError(
            f"Checkpoint file must be named model_XXXXXX.pt, got: {model_path}"
        )
    return int(match.group(1))


def _load_model_from_checkpoint_path(checkpoint: str, device, phase: str):
    checkpoint = os.path.abspath(checkpoint)
    if os.path.isdir(checkpoint):
        checkpoint_dir = checkpoint
        step = find_last_step(checkpoint_dir)
    else:
        checkpoint_dir = os.path.dirname(checkpoint)
        step = _parse_step_from_model_file(checkpoint)
    return build_model(checkpoint_dir, step, device, phase=phase)


def _answer_question(
    question: str,
    conversation_tokens: list[int],
    tokenizer,
    engine,
    autocast_ctx,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    user_start: int,
    user_end: int,
    assistant_start: int,
    assistant_end: int,
) -> tuple[list[int], str]:
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(question))
    conversation_tokens.append(user_end)
    conversation_tokens.append(assistant_start)

    generated = []
    answer_chunks = []
    print("\nAssistant: ", end="", flush=True)
    with autocast_ctx:
        for token_column, _ in engine.generate(
            conversation_tokens,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        ):
            token = token_column[0]
            generated.append(token)
            token_text = tokenizer.decode([token])
            answer_chunks.append(token_text)
            print(token_text, end="", flush=True)
    print()

    if not generated or generated[-1] != assistant_end:
        generated.append(assistant_end)
    conversation_tokens.extend(generated)
    return conversation_tokens, "".join(answer_chunks).strip()


def _load_questions(questions_file: str) -> list[str]:
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions


def main():
    parser = argparse.ArgumentParser(description="Chat with a specific nanochat checkpoint")
    parser.add_argument(
        "--source",
        type=str,
        default="sft",
        choices=["base", "sft", "rl"],
        help="Checkpoint source directory when --checkpoint is not set",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="Model tag to load (e.g. parallel-24)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to load (defaults to latest)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit checkpoint directory or model_XXXXXX.pt path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Single prompt mode (prints one response and exits)",
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default="",
        help="If provided, answer each non-empty line in this file sequentially",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16"],
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="",
        choices=["cuda", "cpu", "mps"],
        help="cuda|cpu|mps (empty = autodetect)",
    )
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )

    if args.checkpoint:
        model, tokenizer, meta = _load_model_from_checkpoint_path(
            args.checkpoint, device, phase="eval"
        )
    else:
        model, tokenizer, meta = load_model(
            args.source,
            device,
            phase="eval",
            model_tag=args.model_tag,
            step=args.step,
        )

    model_tag = args.model_tag
    step = meta.get("step", "<unknown>")
    print(f"Loaded model: tag={model_tag}, step={step}, device={device}")

    base_dir = os.environ.get("NANOCHAT_BASE_DIR", "/vol/.cache/nanochat")
    safe_model_tag = str(model_tag) if model_tag else "unknown-model"
    answer_dir = os.path.join(base_dir, "qa_results", safe_model_tag)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, "answer.txt")
    print(f"Writing answers to: {answer_path}")

    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    engine = Engine(model, tokenizer)

    print("\nNanoChat test console")
    print("-" * 50)
    print("Commands: 'quit'/'exit' to stop, 'clear' to reset conversation.")
    print("-" * 50)

    conversation_tokens = [bos]

    with open(answer_path, "w", encoding="utf-8") as answer_file:
        answer_file.write(f"Model tag: {model_tag}\n")
        answer_file.write(f"Step: {step}\n")
        answer_file.write(f"Source: {args.source}\n")
        answer_file.write(f"Checkpoint: {args.checkpoint or '<auto>'}\n\n")

        if args.questions_file:
            if not os.path.exists(args.questions_file):
                raise FileNotFoundError(
                    f"Questions file not found: {args.questions_file}"
                )
            questions = _load_questions(args.questions_file)
            if not questions:
                print(f"No questions found in {args.questions_file}")
                compute_cleanup()
                return

            print(f"Batch mode: answering {len(questions)} questions from {args.questions_file}")
            for idx, question in enumerate(questions, start=1):
                print(f"\nQuestion {idx}: {question}")
                conversation_tokens = [bos]
                conversation_tokens, answer = _answer_question(
                    question=question,
                    conversation_tokens=conversation_tokens,
                    tokenizer=tokenizer,
                    engine=engine,
                    autocast_ctx=autocast_ctx,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    user_start=user_start,
                    user_end=user_end,
                    assistant_start=assistant_start,
                    assistant_end=assistant_end,
                )
                answer_file.write(f"Question {idx}: {question}\n")
                answer_file.write(f"Assistant: {answer}\n\n")
            compute_cleanup()
            return

        turn = 0
        while True:
            if args.prompt:
                user_input = args.prompt.strip()
            else:
                try:
                    user_input = input("\nUser: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

            if user_input.lower() in {"quit", "exit"}:
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                conversation_tokens = [bos]
                print("Conversation cleared.")
                continue
            if not user_input:
                continue

            conversation_tokens, answer = _answer_question(
                question=user_input,
                conversation_tokens=conversation_tokens,
                tokenizer=tokenizer,
                engine=engine,
                autocast_ctx=autocast_ctx,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                user_start=user_start,
                user_end=user_end,
                assistant_start=assistant_start,
                assistant_end=assistant_end,
            )
            turn += 1
            answer_file.write(f"Question {turn}: {user_input}\n")
            answer_file.write(f"Assistant: {answer}\n\n")
            answer_file.flush()

            if args.prompt:
                break

    compute_cleanup()


if __name__ == "__main__":
    main()
