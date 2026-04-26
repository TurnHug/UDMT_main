import argparse
import json
import os
import time

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from config import TrainConfig
from model import build_sod_model, get_saliency_probs
from utils import ensure_dir, read_last_exp_dir

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on a test set and save saliency predictions."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="",
        help="Direct dataset path containing image/input/images and optional GT/mask folders.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test_real",
        help="Dataset split name under cfg.data_root when --data-path is not provided.",
    )
    parser.add_argument("--experiments-dir", dest="exp_dir", type=str, default="experiments/ssod_20260418_112748")
    parser.add_argument("--experiments-root", dest="exp_root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="predict/CSOD10K",
        help="Directory to save prediction maps. Defaults to <exp_dir>/predictions/<dataset_name>.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def load_config(exp_dir):
    if exp_dir is None:
        return TrainConfig()

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

    if exp_dir is None:
        raise FileNotFoundError(
            "No checkpoint was provided. Pass --checkpoint or --experiments-dir."
        )

    candidates = [
        os.path.join(exp_dir, "checkpoints", "teacher_best_mae.pth"),
        os.path.join(exp_dir, "checkpoints", "teacher_latest.pth"),
        os.path.join(exp_dir, "checkpoints", "student_best_mae.pth"),
        os.path.join(exp_dir, "checkpoints", "student_latest.pth"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No usable checkpoint found under: {exp_dir}")


def load_model_state(model, checkpoint_path, device):
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


def _find_existing_dir(candidates):
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def _index_image_files(folder):
    out = {}
    for filename in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(filename)
        if ext.lower() in _IMG_EXTS:
            out[stem] = filename
    return out


def resolve_dataset_dirs(dataset_root):
    image_dir = _find_existing_dir(
        [
            os.path.join(dataset_root, "input"),
            os.path.join(dataset_root, "images"),
            os.path.join(dataset_root, "image"),
            os.path.join(dataset_root, "imgs"),
        ]
    )
    if image_dir is None:
        raise FileNotFoundError(
            "Could not find an image directory under dataset root. "
            "Expected one of: input, images, image, imgs."
        )

    gt_dir = _find_existing_dir(
        [
            os.path.join(dataset_root, "gt"),
            os.path.join(dataset_root, "GT"),
            os.path.join(dataset_root, "mask"),
            os.path.join(dataset_root, "masks"),
        ]
    )
    return image_dir, gt_dir


class InferenceDataset(Dataset):
    def __init__(self, dataset_root, image_size, norm_mean, norm_std):
        self.dataset_root = dataset_root
        self.image_dir, self.gt_dir = resolve_dataset_dirs(dataset_root)
        self.image_map = _index_image_files(self.image_dir)
        if not self.image_map:
            raise FileNotFoundError(f"No images found in: {self.image_dir}")

        self.gt_map = _index_image_files(self.gt_dir) if self.gt_dir else {}
        self.sample_ids = sorted(self.image_map.keys())
        self.normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        image_path = os.path.join(self.image_dir, self.image_map[sample_id])
        with Image.open(image_path) as im:
            image = im.convert("RGB")
            original_width, original_height = image.size
            image_tensor = self.normalize(TF.to_tensor(image))

        target_height, target_width = original_height, original_width
        if sample_id in self.gt_map:
            gt_path = os.path.join(self.gt_dir, self.gt_map[sample_id])
            with Image.open(gt_path) as gm:
                gt = gm.convert("L")
                target_width, target_height = gt.size

        filename = os.path.splitext(self.image_map[sample_id])[0] + ".png"
        return image_tensor, target_height, target_width, filename


def build_loader(dataset_root, cfg, batch_size):
    dataset = InferenceDataset(
        dataset_root=dataset_root,
        image_size=cfg.image_size,
        norm_mean=list(cfg.norm_mean),
        norm_std=list(cfg.norm_std),
    )
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
    )


def resolve_dataset_root(args, cfg):
    if args.data_path:
        dataset_root = args.data_path
        dataset_name = os.path.basename(os.path.normpath(dataset_root))
        return dataset_root, dataset_name

    dataset_root = os.path.join(cfg.data_root, args.split)
    dataset_name = args.split
    return dataset_root, dataset_name


def default_save_dir(args, exp_dir, dataset_name):
    if args.save_dir:
        return args.save_dir
    if exp_dir is not None:
        return os.path.join(exp_dir, "predictions", dataset_name)
    return os.path.join("predictions", dataset_name)


def save_predictions(model, loader, device, save_dir):
    ensure_dir(save_dir)
    total_time = 0.0
    total_images = 0

    model.eval()
    with torch.no_grad():
        for images, heights, widths, filenames in tqdm(loader, desc="test", leave=False):
            images = images.to(device, non_blocking=True)

            time_start = time.time()
            probs = get_saliency_probs(model(images))
            total_time += time.time() - time_start

            total_images += images.shape[0]
            for batch_idx in range(images.shape[0]):
                pred = probs[batch_idx : batch_idx + 1]
                pred = torch.nn.functional.interpolate(
                    pred,
                    size=(int(heights[batch_idx]), int(widths[batch_idx])),
                    mode="bilinear",
                    align_corners=False,
                )
                pred_np = pred[0, 0].float().cpu().numpy()
                pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-10)
                pred_u8 = np.clip(pred_np * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
                imageio.imwrite(os.path.join(save_dir, filenames[batch_idx]), pred_u8)

    avg_time = total_time / max(1, total_images)
    fps = total_images / max(total_time, 1e-12)
    return total_images, avg_time, fps


def main():
    args = parse_args()
    exp_root = args.exp_root or TrainConfig().exp_root
    exp_dir = args.exp_dir
    if exp_dir is None and args.checkpoint is None:
        try:
            exp_dir = read_last_exp_dir(exp_root)
        except FileNotFoundError:
            exp_dir = None

    cfg = load_config(exp_dir)
    checkpoint_path = resolve_checkpoint(exp_dir, args.checkpoint)
    dataset_root, dataset_name = resolve_dataset_root(args, cfg)
    save_dir = default_save_dir(args, exp_dir, dataset_name)

    device = torch.device(args.device) if args.device else torch.device(
        cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = build_sod_model(cfg).to(device)
    load_model_state(model, checkpoint_path, device)

    loader = build_loader(dataset_root, cfg, batch_size=args.batch_size)
    num_images, avg_time, fps = save_predictions(model, loader, device, save_dir)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Saved predictions: {save_dir}")
    print(f"Processed images: {num_images}")
    print(f"Running time {avg_time:.5f} s / image")
    print(f"Average speed: {fps:.4f} fps")


if __name__ == "__main__":
    main()
