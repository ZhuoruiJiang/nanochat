"""
The CodeAlpaca instruction-following coding dataset.
https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K
"""

from datasets import load_dataset
from tasks.common import Task


def render_codealpaca_prompt(prompt):
    """Render the CodeAlpaca user prompt."""
    return prompt.strip()


class CodeAlpaca(Task):
    """CodeAlpaca SFT dataset using the full train split."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split == "train", "CodeAlpaca only provides the train split"
        self.split = split
        self.ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split=split)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        user_message = render_codealpaca_prompt(row["prompt"])
        assistant_message = row["completion"]
        conversation = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ],
        }
        return conversation
