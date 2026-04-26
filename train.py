import argparse
import json
import os
from itertools import cycle

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from config import TrainConfig
from data import list_paired_samples, make_eval_loader, make_labeled_loader, make_unlabeled_loader, split_labeled_unlabeled
from losses import edl_loss, uncertainty_calibration_loss
from model.sod_net import build_sod_model, get_beta_params, get_saliency_probs, get_uncertainty_map
from utils import count_parameters, ensure_dir, make_exp_dir, save_file_lists, save_json, set_seed, sigmoid_rampup, write_last_exp_dir

try:
    mp.set_sharing_strategy("file_system")
except (AttributeError, RuntimeError):
    pass


def _parse_bool(text):
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {text}")


def _coerce_cli_value(raw_value, default_value):
    if isinstance(default_value, bool):
        return _parse_bool(raw_value)
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    if isinstance(default_value, list):
        return json.loads(raw_value)
    return raw_value


def build_cfg_from_cli():
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train the Uncertainty-Driven Mean Teacher Framework for semi-supervised SOD"
    )
    for key in cfg.to_dict():
        parser.add_argument(f"--{key}", default=None, type=str)
    args = parser.parse_args()

    for key, raw_value in vars(args).items():
        if raw_value is None or not hasattr(cfg, key):
            continue
        default_value = getattr(cfg, key)
        setattr(cfg, key, _coerce_cli_value(raw_value, default_value))
    return cfg


def create_grad_scaler(device, enabled):
    amp_enabled = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def select_device(cfg):
    if cfg.device:
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log_epoch(log_path, row):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_text_log(log_path, message):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def copy_student_to_teacher(student, teacher):
    teacher.load_state_dict(student.state_dict())
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)


class ReliabilityEMAController:
    """
    Innovation 2: Uncertainty-Aware Exponential Moving Average.

    The teacher update speed is controlled by the uncertainty-derived
    quality of the current unlabeled batch instead of a fixed EMA rule.
    """

    def __init__(self, quality_momentum, quality_gamma, teacher_ema_min, teacher_ema_max):
        self.quality_momentum = float(quality_momentum)
        self.quality_gamma = float(quality_gamma)
        self.teacher_ema_min = float(teacher_ema_min)
        self.teacher_ema_max = float(teacher_ema_max)
        self.smoothed_quality = None

    def update(self, batch_quality):
        quality = max(0.0, min(1.0, float(batch_quality)))
        if self.smoothed_quality is None:
            self.smoothed_quality = quality
        else:
            momentum = self.quality_momentum
            self.smoothed_quality = momentum * self.smoothed_quality + (1.0 - momentum) * quality

        # Higher uncertainty-derived quality means the teacher can absorb
        # student knowledge faster; lower quality makes the update more conservative.
        quality_bar = max(0.0, min(1.0, float(self.smoothed_quality)))
        quality_power = quality_bar ** self.quality_gamma
        teacher_decay = self.teacher_ema_max - (self.teacher_ema_max - self.teacher_ema_min) * quality_power
        return quality_bar, teacher_decay

    def state_dict(self):
        return {
            "quality_momentum": self.quality_momentum,
            "quality_gamma": self.quality_gamma,
            "teacher_ema_min": self.teacher_ema_min,
            "teacher_ema_max": self.teacher_ema_max,
            "smoothed_quality": self.smoothed_quality,
        }


@torch.no_grad()
def update_teacher_weights(student, teacher, decay):
    for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
        teacher_parameter.data.mul_(decay).add_(student_parameter.data, alpha=1.0 - decay)

    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)


def current_unsup_weight(cfg, epoch, has_unlabeled):
    if not has_unlabeled or epoch <= int(cfg.supervised_only_epochs):
        return 0.0
    ramp_epoch = epoch - int(cfg.supervised_only_epochs)
    return float(cfg.unsup_weight_max) * sigmoid_rampup(ramp_epoch, int(cfg.unsup_rampup_epochs))


def save_checkpoint(exp_dir, epoch, student, teacher, optimizer, scheduler, scaler, reliability_ema):
    checkpoint = {
        "epoch": epoch,
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "reliability_ema": reliability_ema.state_dict(),
    }
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    torch.save(checkpoint, os.path.join(ckpt_dir, "checkpoint_latest.pt"))
    torch.save(student.state_dict(), os.path.join(ckpt_dir, "student_latest.pth"))
    torch.save(teacher.state_dict(), os.path.join(ckpt_dir, "teacher_latest.pth"))
    torch.save(checkpoint, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))




def train_one_epoch(
    cfg,
    epoch,
    student,
    teacher,
    labeled_loader,
    unlabeled_loader,
    optimizer,
    scaler,
    device,
    reliability_ema,
):
    student.train()
    teacher.eval()

    use_amp = bool(cfg.use_amp and device.type == "cuda")
    unlabeled_active = unlabeled_loader is not None
    unsup_weight = current_unsup_weight(cfg, epoch, unlabeled_active)

    labeled_iter = cycle(labeled_loader)
    unlabeled_iter = cycle(unlabeled_loader) if unlabeled_active else None

    steps = len(labeled_loader)
    if unlabeled_active and unsup_weight > 0.0:
        steps = max(steps, len(unlabeled_loader))
    if cfg.steps_per_epoch and cfg.steps_per_epoch > 0:
        steps = min(steps, int(cfg.steps_per_epoch))

    meters = {
        "loss_total": 0.0,
        "loss_supervised": 0.0,
        "loss_unsupervised": 0.0,
        "sup_edl_loss": 0.0,
        "sup_seg_loss": 0.0,
        "sup_nll_loss": 0.0,
        "sup_kl_loss": 0.0,
        "sup_confidence_mean": 0.0,
        "sup_uncertainty_mean": 0.0,
        "unsup_pseudo_edl_loss": 0.0,
        "unsup_pseudo_seg_loss": 0.0,
        "unsup_pseudo_nll_loss": 0.0,
        "unsup_pseudo_kl_loss": 0.0,
        "confidence_weight_mean": 0.0,
        "conf_mask_ratio": 0.0,
        "teacher_pseudo_fg_mean": 0.0,
        "student_uncertainty_mean": 0.0,
        "student_uncertainty_mean_unsup": 0.0,
        "teacher_uncertainty_mean": 0.0,
        "unsup_calib_loss": 0.0,
        "batch_quality": 0.0,
        "quality_ema": 0.0,
        "teacher_ema_decay": 0.0,
    }

    progress = tqdm(range(steps), desc=f"Epoch {epoch}/{cfg.max_epochs}", leave=False)
    for _ in progress:
        labeled_images, labeled_masks, _ = next(labeled_iter)
        labeled_images = labeled_images.to(device, non_blocking=True)
        labeled_masks = labeled_masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            labeled_outputs = student(labeled_images)
            labeled_alpha, labeled_beta = get_beta_params(labeled_outputs)
            labeled_uncertainty = get_uncertainty_map(labeled_outputs)

            loss_sup, supervised_stats = edl_loss(
                alpha=labeled_alpha,
                beta=labeled_beta,
                y_true=labeled_masks,
                lambda_kl=float(cfg.edl_lambda_kl),
            )
            sup_edl_loss = supervised_stats["sup_edl_loss"]
            sup_seg_loss = supervised_stats["sup_seg_loss"]
            sup_nll_loss = supervised_stats["sup_nll_loss"]
            sup_kl_loss = supervised_stats["sup_kl_loss"]
            sup_confidence_mean = supervised_stats["sup_confidence_mean"]
            sup_uncertainty_mean = supervised_stats["sup_uncertainty_mean"]
            student_uncertainty_mean = 0.0 if labeled_uncertainty is None else float(labeled_uncertainty.mean().item())

            loss_unsup = torch.zeros((), device=device)
            unsup_pseudo_edl_loss = 0.0
            unsup_pseudo_seg_loss = 0.0
            unsup_pseudo_nll_loss = 0.0
            unsup_pseudo_kl_loss = 0.0
            confidence_weight_mean = 0.0
            conf_mask_ratio = 0.0
            teacher_pseudo_fg_mean = 0.0
            teacher_uncertainty_mean = 0.0
            student_uncertainty_mean_unsup = 0.0
            unsup_calib_loss = 0.0
            batch_quality = 0.0

            if unlabeled_active and unsup_weight > 0.0:
                weak_images, strong_images, _ = next(unlabeled_iter)
                weak_images = weak_images.to(device, non_blocking=True)
                strong_images = strong_images.to(device, non_blocking=True)

                with torch.no_grad():
                    teacher_outputs = teacher(weak_images)
                    teacher_probs = get_saliency_probs(teacher_outputs)
                    teacher_uncertainty = get_uncertainty_map(teacher_outputs)
                    pseudo_masks = (teacher_probs >= float(cfg.pseudo_label_threshold)).to(labeled_masks.dtype)

                student_outputs_u = student(strong_images)
                student_probs_u = get_saliency_probs(student_outputs_u)
                student_uncertainty_u = get_uncertainty_map(student_outputs_u)
                student_alpha_u, student_beta_u = get_beta_params(student_outputs_u)

                if teacher_uncertainty is None or student_uncertainty_u is None:
                    raise RuntimeError(
                        "The unlabeled branch requires both teacher and student uncertainty maps."
                    )

                loss_unsup_pseudo, unsup_supervised_stats = edl_loss(
                    alpha=student_alpha_u,
                    beta=student_beta_u,
                    y_true=pseudo_masks,
                    lambda_kl=float(cfg.edl_lambda_kl),
                )
                loss_unsup_calib, dual_aux = uncertainty_calibration_loss(
                    student_probs=student_probs_u,
                    student_uncertainty=student_uncertainty_u,
                    teacher_probs=teacher_probs,
                    teacher_uncertainty=teacher_uncertainty,
                    confidence_threshold=float(cfg.dual_ucc_conf_threshold),
                )
                loss_unsup = loss_unsup_pseudo + loss_unsup_calib
                consistency_stats = dual_aux["stats"]
                batch_quality = dual_aux["batch_quality"]

                unsup_pseudo_edl_loss = unsup_supervised_stats["sup_edl_loss"]
                unsup_pseudo_seg_loss = unsup_supervised_stats["sup_seg_loss"]
                unsup_pseudo_nll_loss = unsup_supervised_stats["sup_nll_loss"]
                unsup_pseudo_kl_loss = unsup_supervised_stats["sup_kl_loss"]
                confidence_weight_mean = consistency_stats["confidence_weight_mean"]
                conf_mask_ratio = consistency_stats["conf_mask_ratio"]
                teacher_pseudo_fg_mean = consistency_stats["teacher_pseudo_fg_mean"]
                teacher_uncertainty_mean = consistency_stats["teacher_uncertainty_mean"]
                student_uncertainty_mean_unsup = consistency_stats["student_uncertainty_mean_unsup"]
                unsup_calib_loss = consistency_stats["unsup_calib_loss"]

            loss = loss_sup + unsup_weight * loss_unsup

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg.grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg.grad_clip))
            optimizer.step()

        if unlabeled_active and unsup_weight > 0.0:
            quality_bar, teacher_decay = reliability_ema.update(batch_quality)
        else:
            quality_bar = 0.0 if reliability_ema.smoothed_quality is None else float(reliability_ema.smoothed_quality)
            teacher_decay = float(cfg.teacher_ema_supervised)
        update_teacher_weights(student, teacher, teacher_decay)

        meters["loss_total"] += float(loss.item())
        meters["loss_supervised"] += float(loss_sup.item())
        meters["loss_unsupervised"] += float(loss_unsup.item())
        meters["sup_edl_loss"] += sup_edl_loss
        meters["sup_seg_loss"] += sup_seg_loss
        meters["sup_nll_loss"] += sup_nll_loss
        meters["sup_kl_loss"] += sup_kl_loss
        meters["sup_confidence_mean"] += sup_confidence_mean
        meters["sup_uncertainty_mean"] += sup_uncertainty_mean
        meters["unsup_pseudo_edl_loss"] += unsup_pseudo_edl_loss
        meters["unsup_pseudo_seg_loss"] += unsup_pseudo_seg_loss
        meters["unsup_pseudo_nll_loss"] += unsup_pseudo_nll_loss
        meters["unsup_pseudo_kl_loss"] += unsup_pseudo_kl_loss
        meters["confidence_weight_mean"] += confidence_weight_mean
        meters["conf_mask_ratio"] += conf_mask_ratio
        meters["teacher_pseudo_fg_mean"] += teacher_pseudo_fg_mean
        meters["student_uncertainty_mean"] += student_uncertainty_mean
        meters["student_uncertainty_mean_unsup"] += student_uncertainty_mean_unsup
        meters["teacher_uncertainty_mean"] += teacher_uncertainty_mean
        meters["unsup_calib_loss"] += unsup_calib_loss
        meters["batch_quality"] += batch_quality
        meters["quality_ema"] += quality_bar
        meters["teacher_ema_decay"] += teacher_decay

        progress.set_postfix(
            total=f"{loss.item():.3f}",
            sup=f"{loss_sup.item():.3f}",
            edl=f"{sup_edl_loss:.3f}",
            unsup=f"{loss_unsup.item():.3f}",
            q=f"{batch_quality:.3f}",
            qbar=f"{quality_bar:.3f}",
            ema=f"{teacher_decay:.4f}",
        )

    for key in meters:
        meters[key] /= max(1, steps)
    meters["unsup_weight"] = unsup_weight
    return meters


def train(cfg=None):
    cfg = cfg if cfg is not None else TrainConfig()
    set_seed(int(cfg.split_seed))

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    exp_dir = make_exp_dir(cfg.exp_root, cfg.exp_name, cfg.labeled_ratio)
    write_last_exp_dir(cfg.exp_root, exp_dir)
    save_json(os.path.join(exp_dir, "logs", "config.json"), cfg.to_dict())

    all_files = list_paired_samples(cfg.data_root, cfg.train_split)
    labeled_files, unlabeled_files = split_labeled_unlabeled(all_files, cfg.labeled_ratio, cfg.split_seed)
    if cfg.save_split_list:
        save_file_lists(
            os.path.join(exp_dir, "split_labeled_unlabeled.json"),
            labeled_files,
            unlabeled_files,
            cfg.split_seed,
            cfg.labeled_ratio,
        )

    labeled_loader = make_labeled_loader(cfg, labeled_files)
    unlabeled_loader = make_unlabeled_loader(cfg, unlabeled_files)
    test_real_loader = make_eval_loader(cfg, "test_real")

    device = select_device(cfg)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    student = build_sod_model(cfg).to(device)
    teacher = build_sod_model(cfg).to(device)
    copy_student_to_teacher(student, teacher)

    reliability_ema = ReliabilityEMAController(
        quality_momentum=float(cfg.quality_ema_momentum),
        quality_gamma=float(cfg.quality_gamma),
        teacher_ema_min=float(cfg.teacher_ema_min),
        teacher_ema_max=float(cfg.teacher_ema_max),
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(cfg.max_epochs)),
        eta_min=float(cfg.min_lr),
    )
    scaler = create_grad_scaler(device, enabled=bool(cfg.use_amp))

    ensure_dir(os.path.join(exp_dir, "logs"))
    log_path = os.path.join(exp_dir, "logs", "train.jsonl")
    text_log_path = os.path.join(exp_dir, "logs", "train.log")

    startup_line = (
        f"[train] device={device} "
        f"student_params={count_parameters(student)} "
        f"samples(total/labeled/unlabeled)={len(all_files)}/{len(labeled_files)}/{len(unlabeled_files)}"
    )
    print(startup_line)
    _append_text_log(text_log_path, startup_line)

    exp_line = f"[train] experiment_dir={exp_dir}"
    print(exp_line)
    _append_text_log(text_log_path, exp_line)

    method_identity = cfg.framework_summary()
    framework_line = (
        "[train] "
        f"framework={method_identity['framework_cn']} "
        f"({method_identity['framework_en']})"
    )
    innovation1_line = (
        "[train] "
        f"innovation_1={method_identity['innovation1_cn']} "
        f"({method_identity['innovation1_en']})"
    )
    innovation2_line = (
        "[train] "
        f"innovation_2={method_identity['innovation2_cn']} "
        f"({method_identity['innovation2_en']})"
    )
    print(framework_line)
    print(innovation1_line)
    print(innovation2_line)
    _append_text_log(text_log_path, framework_line)
    _append_text_log(text_log_path, innovation1_line)
    _append_text_log(text_log_path, innovation2_line)

    config_line = (
        "[train] "
        f"edl_lambda_kl={cfg.edl_lambda_kl:.4f} "
        f"pseudo_label_threshold={cfg.pseudo_label_threshold:.2f} "
        f"dual_ucc_conf_threshold={cfg.dual_ucc_conf_threshold:.4f} "
        f"unsup_weight_max={cfg.unsup_weight_max:.2f} "
        f"supervised_only_epochs={cfg.supervised_only_epochs} "
        f"unsup_rampup_epochs={cfg.unsup_rampup_epochs} "
        f"quality_ema_momentum={cfg.quality_ema_momentum:.2f} "
        f"quality_gamma={cfg.quality_gamma:.2f} "
        f"teacher_ema_supervised={cfg.teacher_ema_supervised:.4f} "
        f"teacher_ema_range=({cfg.teacher_ema_min:.4f}, {cfg.teacher_ema_max:.4f})"
    )
    print(config_line)
    _append_text_log(text_log_path, config_line)

    for epoch in range(1, int(cfg.max_epochs) + 1):
        epoch_stats = train_one_epoch(
            cfg=cfg,
            epoch=epoch,
            student=student,
            teacher=teacher,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            reliability_ema=reliability_ema,
        )
        scheduler.step()


  

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **epoch_stats,
        }
        _log_epoch(log_path, row)
        epoch_line = (
            f"Epoch {epoch:03d} | "
            f"Loss {epoch_stats['loss_total']:.4f} | "
            f"Sup {epoch_stats['loss_supervised']:.4f} | "
            f"Unsup {epoch_stats['loss_unsupervised']:.4f} | "
            f"UnsupW {epoch_stats['unsup_weight']:.3f} | "
            f"SupEDL {epoch_stats['sup_edl_loss']:.3f} | "
            f"ConfW {epoch_stats['confidence_weight_mean']:.3f} | "
            f"Mask {epoch_stats['conf_mask_ratio']:.3f} | "
            f"Q {epoch_stats['batch_quality']:.3f} | "
            f"Qbar {epoch_stats['quality_ema']:.3f} | "
            f"TeacherEMA {epoch_stats['teacher_ema_decay']:.4f} | "
            f"TeacherFG {epoch_stats['teacher_pseudo_fg_mean']:.3f} | "
            f"StudentUnc {epoch_stats['student_uncertainty_mean']:.3f} | "
            f"PseudoEDL {epoch_stats['unsup_pseudo_edl_loss']:.3f} | "
            f"StudentUncU {epoch_stats['student_uncertainty_mean_unsup']:.3f} | "
            f"TeacherUnc {epoch_stats['teacher_uncertainty_mean']:.3f} | "
            f"Calib {epoch_stats['unsup_calib_loss']:.3f}"
        )

        print(epoch_line)
        _append_text_log(text_log_path, epoch_line)

        if epoch % int(cfg.save_every) == 0 or epoch == int(cfg.max_epochs):
            save_checkpoint(
                exp_dir,
                epoch,
                student,
                teacher,
                optimizer,
                scheduler,
                scaler,
                reliability_ema,
            )

    finish_line = f"Training finished. Experiment dir: {exp_dir}"
    print(finish_line)
    _append_text_log(text_log_path, finish_line)


if __name__ == "__main__":
    train(build_cfg_from_cli())
