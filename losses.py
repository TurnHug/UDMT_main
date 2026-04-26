import torch
import torch.nn.functional as F


def _soft_iou_loss(probs, targets, eps=1e-6):
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return 1.0 - (intersection + eps) / (union + eps)


def kl_beta(alpha, beta):
    """
    KL[ Beta(alpha, beta) || Beta(1, 1) ]
    """
    alpha = alpha.clamp_min(1e-6)
    beta = beta.clamp_min(1e-6)
    S = alpha + beta
    loss = (
        torch.lgamma(S)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        - (S - 2.0) * torch.digamma(S)
        + (alpha - 1.0) * torch.digamma(alpha)
        + (beta - 1.0) * torch.digamma(beta)
    )
    return loss.mean()


def edl_loss(alpha, beta, y_true, lambda_kl=0.1, eps=1e-6):
    """
    Labeled branch supervision with Beta evidence.

    The model predicts evidence for foreground / background, which induces
    a Beta distribution Beta(alpha, beta). We supervise the labeled branch
    directly in the evidence space and keep the uncertainty-aware
    segmentation weighting requested by the user.
    """
    alpha = alpha.clamp_min(1.0 + eps)
    beta = beta.clamp_min(1.0 + eps)
    y_true = y_true.to(dtype=alpha.dtype)

    S = alpha + beta
    pred = (alpha / S).clamp(min=eps, max=1.0 - eps)

    loss_nll = y_true * (torch.log(S) - torch.log(alpha)) + (1.0 - y_true) * (torch.log(S) - torch.log(beta))
    loss_nll = loss_nll.mean()

    alpha_tilde = y_true * (1.0 + beta) + (1.0 - y_true) * alpha
    beta_tilde = y_true * beta + (1.0 - y_true) * (1.0 + alpha)
    loss_kl = kl_beta(alpha_tilde, beta_tilde)

    uncertainty = (2.0 / S).detach()
    confidence = (1.0 - uncertainty).clamp(min=0.0, max=1.0)

    bce_map = F.binary_cross_entropy(pred, y_true, reduction="none")
    bce_loss = (confidence * bce_map).mean()

    iou_per_sample = _soft_iou_loss(pred, y_true, eps=eps)
    confidence_per_sample = confidence.mean(dim=(1, 2, 3))
    iou_loss = (confidence_per_sample * iou_per_sample).mean()

    loss_seg = bce_loss + iou_loss
    total_loss = loss_seg + loss_nll + float(lambda_kl) * loss_kl

    stats = {
        "sup_edl_loss": float(total_loss.item()),
        "sup_seg_loss": float(loss_seg.item()),
        "sup_nll_loss": float(loss_nll.item()),
        "sup_kl_loss": float(loss_kl.item()),
        "sup_confidence_mean": float(confidence.mean().item()),
        "sup_uncertainty_mean": float((2.0 / S).mean().item()),
    }
    return total_loss, stats


def _masked_average(loss_map, mask, eps=1e-6):
    weighted_sum = (loss_map * mask).sum()
    normalizer = mask.sum().clamp_min(eps)
    return weighted_sum / normalizer


def uncertainty_calibration_loss(
    student_probs,
    student_uncertainty,
    teacher_probs,
    teacher_uncertainty,
    confidence_threshold=0.01,
    eps=1e-6,
):
    """
    Uncertainty calibration on the unlabeled branch.

    The student uncertainty should explain its deviation from the teacher
    pseudo prediction, but only in the teacher's high-confidence region.
    """
    teacher_probs = teacher_probs.detach().clamp(min=eps, max=1.0 - eps)
    teacher_uncertainty = teacher_uncertainty.detach().clamp_min(eps)
    student_probs = student_probs.clamp(min=eps, max=1.0 - eps)
    student_uncertainty = student_uncertainty.clamp_min(eps)

    teacher_var = teacher_uncertainty.square()
    student_var = student_uncertainty.square()

    mean_gap_sq = (teacher_probs - student_probs).square()
    confidence_weight = torch.exp(-teacher_var)

    confidence_mask = (teacher_var < float(confidence_threshold)).to(student_probs.dtype)
    calib_loss_map = mean_gap_sq / student_var.clamp_min(eps) + torch.log1p(student_var)
    calib_loss = _masked_average(calib_loss_map, confidence_mask, eps=eps)

    batch_quality = float(confidence_weight.mean().item())
    aux = {
        "confidence_weight": confidence_weight,
        "batch_quality": batch_quality,
        "stats": {
            "confidence_weight_mean": batch_quality,
            "conf_mask_ratio": float(confidence_mask.mean().item()),
            "teacher_uncertainty_mean": float(teacher_uncertainty.mean().item()),
            "student_uncertainty_mean_unsup": float(student_uncertainty.mean().item()),
            "teacher_pseudo_fg_mean": float(teacher_probs.mean().item()),
            "unsup_calib_loss": float(calib_loss.item()),
        },
    }
    return calib_loss, aux
