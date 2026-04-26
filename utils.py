import json
import math
import os
import random
from datetime import datetime

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_exp_dir(exp_root, exp_name, labeled_ratio):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = exp_name if exp_name else f"exp_r{labeled_ratio:g}"
    out_dir = os.path.join(exp_root, f"{prefix}_{ts}")
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "checkpoints"))
    ensure_dir(os.path.join(out_dir, "logs"))
    return out_dir


def write_last_exp_dir(exp_root, exp_dir):
    ensure_dir(exp_root)
    with open(os.path.join(exp_root, "last_exp_dir.txt"), "w", encoding="utf-8") as f:
        f.write(exp_dir)


def read_last_exp_dir(exp_root):
    with open(os.path.join(exp_root, "last_exp_dir.txt"), "r", encoding="utf-8") as f:
        return f.read().strip()


def save_json(path, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_file_lists(path, labeled, unlabeled, seed, ratio):
    save_json(
        path,
        {
            "split_seed": seed,
            "labeled_ratio": ratio,
            "n_labeled": len(labeled),
            "n_unlabeled": len(unlabeled),
            "labeled": sorted(labeled),
            "unlabeled": sorted(unlabeled),
        },
    )


def sigmoid_rampup(current, rampup_length):
    if rampup_length <= 0:
        return 1.0
    current = max(0, min(current, rampup_length))
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())
