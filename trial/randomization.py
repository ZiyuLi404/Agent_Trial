import random

_block = []


def assign_arm(case_id):
    """1:1 block randomization. Returns 'control' or 'treatment'."""
    global _block
    if not _block:
        _block = ["control", "treatment"]
        random.shuffle(_block)
    return _block.pop()
