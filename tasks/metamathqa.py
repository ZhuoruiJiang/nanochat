"""
The MetaMathQA math instruction-following dataset.
https://huggingface.co/datasets/meta-math/MetaMathQA
"""

from datasets import load_dataset
from tasks.common import Task


METAMATHQA_SAMPLE_SIZE = 50_000
METAMATHQA_SAMPLE_SEED = 42


class MetaMathQA(Task):
    """MetaMathQA SFT dataset with deterministic 80K train subsampling."""

    def __init__(self, split="train", sample_size=METAMATHQA_SAMPLE_SIZE, sample_seed=METAMATHQA_SAMPLE_SEED, **kwargs):
        super().__init__(**kwargs)
        assert split == "train", "MetaMathQA only provides the train split for this task"
        assert sample_size > 0, f"sample_size must be positive, got {sample_size}"
        self.split = split
        self.sample_size = sample_size
        self.sample_seed = sample_seed

        ds = load_dataset("meta-math/MetaMathQA", split=split)
        assert sample_size <= len(ds), f"sample_size={sample_size} exceeds dataset size of {len(ds)}"
        self.ds = ds.shuffle(seed=sample_seed).select(range(sample_size))
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        conversation = {
            "messages": [
                {"role": "user", "content": row["query"]},
                {"role": "assistant", "content": row["response"]},
            ],
        }
        return conversation
