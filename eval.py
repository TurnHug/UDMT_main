import argparse
import json
import os

import torch
from tqdm import tqdm

from config import TrainConfig
from data import make_eval_loader
from metrics import RunningSODMetrics
from model.sod_net import build_sod_model, get_saliency_probs
from utils import read_last_exp_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a semi-supervised SOD checkpoint")
    parser.add_argument("--experiments-dir", dest="exp_dir")
    parser.add_argument("--experiments-root", dest="exp_root")
    parser.add_argument("--checkpoint", dest="checkpoint")
    parser.add_argument("--device", dest="device")
    return parser.parse_args()


def load_config(exp_dir):
    config_path = os.path.join(exp_dir, "logs", "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return TrainConfig.from_dict(json.load(f))
    return TrainConfig()


def resolve_checkpoint(exp_dir, explicit_path):
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(f"Checkpoint not found: {explicit_path}")
        return explicit_path

    candidates = [
        os.path.join(exp_dir, "checkpoints", "teacher_latest.pth"),
        os.path.join(exp_dir, "checkpoints", "student_latest.pth"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No evaluation checkpoint found under {exp_dir}/checkpoints")


def load_model_state_strict(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "teacher" in checkpoint:
        state_dict = checkpoint["teacher"]
    elif isinstance(checkpoint, dict) and "student" in checkpoint:
        state_dict = checkpoint["student"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state_dict)}")

    model.load_state_dict(state_dict, strict=True)


@torch.no_grad()
def predict_probs(model, images):
    return get_saliency_probs(model(images))


@torch.no_grad()
def evaluate_split(cfg, model, split, device):
    loader = make_eval_loader(cfg, split)
    meter = RunningSODMetrics()

    for images, masks, _filenames in tqdm(loader, desc=f"eval {split}", leave=False):
        images = images.to(device, non_blocking=True)
        probs = predict_probs(model, images)
        probs = torch.nn.functional.interpolate(probs, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        pred_np = probs[0, 0].float().cpu().numpy()
        gt_np = masks[0, 0].float().cpu().numpy()
        meter.update(pred_np, gt_np)

    return meter.compute()


def main():
    args = parse_args()
    exp_root = args.exp_root or TrainConfig().exp_root
    exp_dir = args.exp_dir or read_last_exp_dir(exp_root)
    cfg = load_config(exp_dir)

    device = torch.device(args.device) if args.device else torch.device(
        cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_path = resolve_checkpoint(exp_dir, args.checkpoint)

    model = build_sod_model(cfg).to(device)
    load_model_state_strict(model, checkpoint_path, device)
    model.eval()

    results = {}
    for split in cfg.test_splits:
        results[split] = evaluate_split(cfg, model, split, device)

    output = {"checkpoint": checkpoint_path, "splits": results}
    save_json(os.path.join(exp_dir, "logs", "eval_results.json"), output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
