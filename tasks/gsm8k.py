"""
GSM8K evaluation.
https://huggingface.co/datasets/openai/gsm8k

Example problem instance:

Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Notice that GSM8K uses tool calls inside << >> tags.
"""

import re
from datasets import load_dataset
from tasks.common import Task


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    Follows official code for normalization:
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class GSM8K(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        question = row['question'] # string of the question prompt
        answer = row['answer'] # string of the full solution and the answer after #### marker
        # Create and return the Conversation object
        # This is tricky because GSM8K uses tool calls, which we need to parse here.
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                # This is a calculator tool call
                inner = part[2:-2]  # Remove << >>
                # Split on = to get expression and result
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                # Add the tool call as a part
                assistant_message_parts.append({"type": "python", "text": expr})
                # Add the result as a part
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                # Regular text in between tool calls
                assistant_message_parts.append({"type": "text", "text": part})
        # Now put it all together
        messages = [
            {"role": "user", "content": question}, # note: simple string
            {"role": "assistant", "content": assistant_message_parts}, # note: list of parts (as dicts)
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Note that:
        - the conversation has both user AND assistant message (containing the ground truth answer)
        - the assistant_response is usually the alternative assistant message achieved via sampling

        TODO: Technically, assistant_response should be a Message (either a string or a list of parts)
              We can handle this later possibly. For now just assume string.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        last_text_part = assistant_message['content'][-1]['text'] # this contains the final answer in GSM8K
        # Extract both the ground truth answer and the predicted answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # Compare and return the success as int
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response, reward_type="binary"):
        """
        Used during RL.

        reward_type:
          - "binary": 1.0 if correct, 0.0 otherwise (original behavior)
          - "distance": continuous reward based on numeric closeness to the ground truth
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        assistant_message = conversation['messages'][-1]
        last_text_part = assistant_message['content'][-1]['text']
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)

        if reward_type == "binary":
            return float(pred_num == ref_num)

        elif reward_type == "distance":
            if pred_num is None:
                return 0.0  # no parseable answer
            if ref_num is None:
                return 0.0  # shouldn't happen, but be safe
            try:
                ref_val = float(ref_num)
                pred_val = float(pred_num)
            except ValueError:
                return 0.0
            if ref_val == pred_val:
                return 1.0
            # Reward = 1 / (1 + relative_error), clamped to [0, 1]
            denom = max(abs(ref_val), 1.0)  # avoid div-by-zero when ref is 0
            relative_error = abs(pred_val - ref_val) / denom
            return 1.0 / (1.0 + relative_error)

        elif reward_type == "reasoning":
            # A. Core accuracy reward
            if pred_num is not None and ref_num is not None and pred_num == ref_num:
                score = 1.0
            else:
                score = -1.0

            # B. Format & termination reward (anti-degeneration)
            has_format = bool(GSM_RE.search(assistant_response))
            if has_format:
                score += 0.2
            else:
                score -= 1.0  # heavy penalty for degeneration / hitting max tokens

            # C. N-gram repetition penalty
            words = assistant_response.split()
            if len(words) >= 5:
                max_run = 1
                current_run = 1
                for i in range(1, len(words)):
                    if words[i].lower() == words[i - 1].lower():
                        current_run += 1
                        max_run = max(max_run, current_run)
                    else:
                        current_run = 1
                if max_run >= 5:
                    score -= 0.5

            # D. Parsimony / length penalty
            num_tokens = len(assistant_response.split())
            score -= 0.001 * num_tokens

            return score
        
        elif reward_type == "hallucination":
            if pred_num is not None and ref_num is not None and pred_num == ref_num:
                score = 1.0
            else:
                score = -1.0
            
            # Hallucinated operand penalty (grounding reward)
            # Seed the available number pool from the user's question
            NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')
            user_question = conversation['messages'][0]['content']
            available_numbers = set(NUM_RE.findall(user_question))

            # Collect all <|python_start|> and <|output_start|> blocks with positions
            all_blocks = []
            for m in re.finditer(r'<\|python_start\|>(.*?)<\|python_end\|>', assistant_response, re.DOTALL):
                all_blocks.append(('python', m.start(), m.group(1)))
            for m in re.finditer(r'<\|output_start\|>(.*?)<\|output_end\|>', assistant_response, re.DOTALL):
                all_blocks.append(('output', m.start(), m.group(1)))

            # Process blocks sequentially by position in the response
            for block_type, _, content in sorted(all_blocks, key=lambda x: x[1]):
                if block_type == 'output':
                    # A completed calculation's result becomes available for future steps
                    available_numbers.update(NUM_RE.findall(content))
                elif block_type == 'python':
                    # Penalise every number in the expression not traceable to prompt or prior outputs
                    for num in NUM_RE.findall(content):
                        if num not in available_numbers:
                            score -= 0.5

            return score
        
        elif reward_type == "combined":
            # A. Accuracy: distance-based continuous score mapped to [-1.0, +1.0]
            #    (upgrades binary ±1.0 from reasoning/reasoning2 with smooth gradient signal)
            if pred_num is None or ref_num is None:
                score = -1.0
            else:
                try:
                    ref_val = float(ref_num)
                    pred_val = float(pred_num)
                except ValueError:
                    score = -1.0
                else:
                    if ref_val == pred_val:
                        distance_score = 1.0
                    else:
                        denom = max(abs(ref_val), 1.0)
                        relative_error = abs(pred_val - ref_val) / denom
                        distance_score = 1.0 / (1.0 + relative_error)
                    score = 2.0 * distance_score - 1.0  # map [0, 1] → [-1, +1]

            # B. Format & termination reward (anti-degeneration)
            has_format = bool(GSM_RE.search(assistant_response))
            if has_format:
                score += 0.2
            else:
                score -= 1.0

            # C. N-gram repetition penalty
            words = assistant_response.split()
            if len(words) >= 5:
                max_run = 1
                current_run = 1
                for i in range(1, len(words)):
                    if words[i].lower() == words[i - 1].lower():
                        current_run += 1
                        max_run = max(max_run, current_run)
                    else:
                        current_run = 1
                if max_run >= 5:
                    score -= 0.5

            # D. Parsimony / length penalty
            score -= 0.001 * len(assistant_response.split())

            # E. Hallucinated operand penalty (grounding reward)
            NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')
            user_question = conversation['messages'][0]['content']
            available_numbers = set(NUM_RE.findall(user_question))

            all_blocks = []
            for m in re.finditer(r'<\|python_start\|>(.*?)<\|python_end\|>', assistant_response, re.DOTALL):
                all_blocks.append(('python', m.start(), m.group(1)))
            for m in re.finditer(r'<\|output_start\|>(.*?)<\|output_end\|>', assistant_response, re.DOTALL):
                all_blocks.append(('output', m.start(), m.group(1)))

            for block_type, _, content in sorted(all_blocks, key=lambda x: x[1]):
                if block_type == 'output':
                    available_numbers.update(NUM_RE.findall(content))
                elif block_type == 'python':
                    for num in NUM_RE.findall(content):
                        if num not in available_numbers:
                            score -= 0.5

            return score

        else:
            raise ValueError(f"Unknown reward_type: {reward_type!r}")