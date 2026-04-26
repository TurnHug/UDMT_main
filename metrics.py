import numpy as np
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure

HAS_PY_SOD_METRICS = True


class RunningSODMetrics:
    """使用 py_sod_metrics 统一累计 SOD 核心指标。"""

    def __init__(self):
        self.sm = Smeasure()
        self.fm = Fmeasure()
        self.em = Emeasure()
        self.mae = MAE()

    @staticmethod
    def _prepare_arrays(pred, gt):
        pred = np.clip(np.asarray(pred, dtype=np.float32), 0.0, 1.0)
        gt = (np.asarray(gt, dtype=np.float32) >= 0.5).astype(np.uint8)
        pred_u8 = (pred * 255.0 + 0.5).astype(np.uint8)
        gt_u8 = gt * 255
        return pred_u8, gt_u8

    def update(self, pred, gt):
        pred_u8, gt_u8 = self._prepare_arrays(pred, gt)
        self.sm.step(pred=pred_u8, gt=gt_u8)
        self.fm.step(pred=pred_u8, gt=gt_u8)
        self.em.step(pred=pred_u8, gt=gt_u8)
        self.mae.step(pred=pred_u8, gt=gt_u8)

    def compute(self):
        f_curve = np.asarray(self.fm.get_results()["fm"]["curve"], dtype=np.float64)
        e_curve = np.asarray(self.em.get_results()["em"]["curve"], dtype=np.float64)
        return {
            "Sm": float(self.sm.get_results()["sm"]),
            "meanF": float(np.mean(f_curve)),
            "meanE": float(np.mean(e_curve)),
            "MAE": float(self.mae.get_results()["mae"]),
        }
