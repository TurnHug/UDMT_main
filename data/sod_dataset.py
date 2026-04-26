import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _find_existing_dir(candidates):
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError("None of expected directories exist:\n" + "\n".join(candidates))


def _resolve_split_dirs(root, split):
    input_candidates = [
        os.path.join(root, split, "input"),
        os.path.join(root, split, "images"),
        os.path.join(root, split, "image"),
        os.path.join(root, split, "imgs"),
        os.path.join(root, f"{split}-Image"),
        os.path.join(root, f"{split}-Imgs"),
    ]
    gt_candidates = [
        os.path.join(root, split, "gt"),
        os.path.join(root, split, "mask"),
        os.path.join(root, split, "masks"),
        os.path.join(root, split, "GT"),
        os.path.join(root, f"{split}-Mask"),
        os.path.join(root, f"{split}-GT"),
    ]
    return _find_existing_dir(input_candidates), _find_existing_dir(gt_candidates)


def _index_image_files(folder):
    out = {}
    for filename in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(filename)
        if ext.lower() in _IMG_EXTS:
            out[stem] = filename
    return out


def _build_split_index(root, split):
    input_dir, gt_dir = _resolve_split_dirs(root, split)
    input_map = _index_image_files(input_dir)
    gt_map = _index_image_files(gt_dir)
    if not input_map:
        raise FileNotFoundError(f"No images found in input dir: {input_dir}")
    if not gt_map:
        raise FileNotFoundError(f"No masks found in gt dir: {gt_dir}")
    return input_dir, gt_dir, input_map, gt_map


def list_paired_samples(root, split):
    _, _, input_map, gt_map = _build_split_index(root, split)
    paired = sorted(set(input_map.keys()) & set(gt_map.keys()))
    if not paired:
        raise RuntimeError(f"No paired samples found for split={split}")
    return paired


def split_labeled_unlabeled(all_files, ratio, seed):
    rng = random.Random(seed)
    indices = list(range(len(all_files)))
    rng.shuffle(indices)
    n_labeled = max(1, int(round(len(all_files) * ratio)))
    labeled_idx = set(indices[:n_labeled])
    labeled = [all_files[i] for i in range(len(all_files)) if i in labeled_idx]
    unlabeled = [all_files[i] for i in range(len(all_files)) if i not in labeled_idx]
    return labeled, unlabeled


class SODAugment:
    def __init__(self, image_size, norm_mean, norm_std, min_scale, max_scale, hflip_p=0.5):
        self.image_size = image_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.hflip_p = hflip_p
        self.normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
        self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08)

    def _sample_spatial_params(self, img):
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            img, scale=(self.min_scale, self.max_scale), ratio=(0.85, 1.15)
        )
        return {"top": top, "left": left, "height": height, "width": width, "flip": random.random() < self.hflip_p}

    def _apply_spatial_image(self, img, params):
        if params["flip"]:
            img = TF.hflip(img)
        img = TF.crop(img, int(params["top"]), int(params["left"]), int(params["height"]), int(params["width"]))
        return TF.resize(img, [self.image_size, self.image_size], InterpolationMode.BILINEAR)

    def _apply_spatial_mask(self, mask, params):
        if params["flip"]:
            mask = TF.hflip(mask)
        mask = TF.crop(mask, int(params["top"]), int(params["left"]), int(params["height"]), int(params["width"]))
        return TF.resize(mask, [self.image_size, self.image_size], InterpolationMode.NEAREST)

    def _to_image_tensor(self, img):
        return self.normalize(TF.to_tensor(img))

    @staticmethod
    def _to_mask_tensor(mask):
        return (TF.to_tensor(mask) >= 0.5).float()

    @staticmethod
    def _add_gaussian_noise(img):
        arr = np.asarray(img).astype(np.float32)
        sigma = random.uniform(5.0, 20.0)
        arr = np.clip(arr + np.random.normal(0.0, sigma, arr.shape), 0.0, 255.0)
        return Image.fromarray(arr.astype(np.uint8))

    @staticmethod
    def _cutout(img):
        out = img.copy()
        draw = ImageDraw.Draw(out)
        width, height = out.size
        cut_w = max(8, int(width * random.uniform(0.12, 0.28)))
        cut_h = max(8, int(height * random.uniform(0.12, 0.28)))
        x0 = random.randint(0, max(0, width - cut_w))
        y0 = random.randint(0, max(0, height - cut_h))
        fill = tuple(random.randint(96, 160) for _ in range(3))
        draw.rectangle((x0, y0, x0 + cut_w, y0 + cut_h), fill=fill)
        return out

    def _strong_appearance(self, img):
        if random.random() < 0.8:
            img = self.color_jitter(img)
        if random.random() < 0.2:
            img = transforms.RandomGrayscale(p=1.0)(img)
        if random.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))
        if random.random() < 0.3:
            img = self._add_gaussian_noise(img)
        if random.random() < 0.5:
            img = self._cutout(img)
        return img

    def augment_labeled(self, img, mask):
        params = self._sample_spatial_params(img)
        aug_img = self._apply_spatial_image(img, params)
        aug_mask = self._apply_spatial_mask(mask, params)
        return self._to_image_tensor(aug_img), self._to_mask_tensor(aug_mask)

    def augment_unlabeled(self, img):
        params = self._sample_spatial_params(img)
        weak_img = self._apply_spatial_image(img, params)
        strong_img = self._strong_appearance(weak_img.copy())
        return self._to_image_tensor(weak_img), self._to_image_tensor(strong_img)

    def prepare_inference_image(self, img):
        img = TF.resize(img, [self.image_size, self.image_size], InterpolationMode.BILINEAR)
        return self._to_image_tensor(img)


class SODLabeledSet(Dataset):
    def __init__(self, root, split, files, augment):
        self.root = root
        self.split = split
        self.files = files
        self.augment = augment
        self.input_dir, self.gt_dir, self.input_map, self.gt_map = _build_split_index(root, split)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample_id = self.files[idx]
        image_path = os.path.join(self.input_dir, self.input_map[sample_id])
        mask_path = os.path.join(self.gt_dir, self.gt_map[sample_id])
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        with Image.open(mask_path) as gm:
            mask = gm.convert("L")
        image_tensor, mask_tensor = self.augment.augment_labeled(image, mask)
        return image_tensor, mask_tensor, sample_id


class SODUnlabeledSet(Dataset):
    def __init__(self, root, split, files, augment):
        self.files = files
        self.augment = augment
        self.input_dir, _, self.input_map, _ = _build_split_index(root, split)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample_id = self.files[idx]
        image_path = os.path.join(self.input_dir, self.input_map[sample_id])
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        weak_tensor, strong_tensor = self.augment.augment_unlabeled(image)
        return weak_tensor, strong_tensor, sample_id


class SODEvalSet(Dataset):
    def __init__(self, root, split, norm_mean, norm_std):
        self.root = root
        self.split = split
        self.files = list_paired_samples(root, split)
        self.input_dir, self.gt_dir, self.input_map, self.gt_map = _build_split_index(root, split)
        self.tf = transforms.Compose(
            [

                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ]
        )
        self.mask_tf = transforms.Compose(
            [

                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample_id = self.files[idx]
        image_path = os.path.join(self.input_dir, self.input_map[sample_id])
        mask_path = os.path.join(self.gt_dir, self.gt_map[sample_id])
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        with Image.open(mask_path) as gm:
            mask = gm.convert("L")
        return self.tf(image), self.mask_tf(mask), sample_id


def make_augment(cfg):
    return SODAugment(
        image_size=int(cfg.image_size),
        norm_mean=list(cfg.norm_mean),
        norm_std=list(cfg.norm_std),
        min_scale=float(cfg.resize_min_scale),
        max_scale=float(cfg.resize_max_scale),
    )


def _loader_runtime_kwargs(cfg):
    num_workers = int(cfg.num_workers)
    kwargs = {"num_workers": num_workers}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        kwargs["prefetch_factor"] = max(1, int(cfg.prefetch_factor))
    return kwargs


def make_labeled_loader(cfg, labeled_files):
    dataset = SODLabeledSet(cfg.data_root, cfg.train_split, labeled_files, make_augment(cfg))
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        drop_last=len(dataset) >= int(cfg.batch_size),
        pin_memory=bool(cfg.pin_memory),
        **_loader_runtime_kwargs(cfg),
    )


def make_unlabeled_loader(cfg, unlabeled_files):
    if not unlabeled_files:
        return None
    dataset = SODUnlabeledSet(cfg.data_root, cfg.train_split, unlabeled_files, make_augment(cfg))
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        drop_last=len(dataset) >= int(cfg.batch_size),
        pin_memory=bool(cfg.pin_memory),
        **_loader_runtime_kwargs(cfg),
    )


def make_loaders(cfg, labeled_files, unlabeled_files):
    return {"labeled": make_labeled_loader(cfg, labeled_files), "unlabeled": make_unlabeled_loader(cfg, unlabeled_files)}


def make_eval_loader(cfg, split_name):
    dataset = SODEvalSet(cfg.data_root, split_name, list(cfg.norm_mean), list(cfg.norm_std))
    return DataLoader(dataset,batch_size=1,shuffle=False,num_workers=4)
