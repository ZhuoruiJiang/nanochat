"""
The Alpaca instruction-following dataset.
https://huggingface.co/datasets/tatsu-lab/alpaca
"""

from datasets import load_dataset
from tasks.common import Task


ALPACA_SAMPLE_NUMERATOR = 5
ALPACA_SAMPLE_DENOMINATOR = 5
ALPACA_SAMPLE_FRAC = ALPACA_SAMPLE_NUMERATOR / ALPACA_SAMPLE_DENOMINATOR
ALPACA_SAMPLE_SEED = 42


def render_alpaca_prompt(instruction, input_text):
    """Render the user prompt in the canonical Alpaca instruction format."""
    prompt = (
        "Below is an instruction that describes a task"
        + (", paired with an input that provides further context." if input_text else ".")
        + " Write a response that appropriately completes the request.\n\n"
        + f"### Instruction:\n{instruction}\n\n"
    )
    if input_text:
        prompt += f"### Input:\n{input_text}\n\n"
    prompt += "### Response:\n"
    return prompt


class Alpaca(Task):
    """Alpaca SFT dataset with deterministic 40% train subsampling."""

    def __init__(self, split="train", sample_frac=ALPACA_SAMPLE_FRAC, sample_seed=ALPACA_SAMPLE_SEED, **kwargs):
        super().__init__(**kwargs)
        assert split == "train", "Alpaca only provides the train split"
        assert 0.0 < sample_frac <= 1.0, f"sample_frac must be in (0, 1], got {sample_frac}"
        self.split = split
        self.sample_frac = sample_frac
        self.sample_seed = sample_seed

        ds = load_dataset("tatsu-lab/alpaca", split=split)
        if sample_frac == ALPACA_SAMPLE_FRAC:
            sample_size = len(ds) * ALPACA_SAMPLE_NUMERATOR // ALPACA_SAMPLE_DENOMINATOR
        else:
            sample_size = int(len(ds) * sample_frac)
        assert sample_size > 0, f"Sampling fraction {sample_frac} produced an empty dataset"
        self.ds = ds.shuffle(seed=sample_seed).select(range(sample_size))
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        input_text = row["input"] or ""
        user_message = render_alpaca_prompt(row["instruction"], input_text)
        assistant_message = row["output"]
        conversation = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ],
        }
        return conversation
