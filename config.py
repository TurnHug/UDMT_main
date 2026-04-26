import os


class TrainConfig:
    def __init__(self):
        # ========= Framework Identity =========
        self.framework_name_cn = "UDMT"
        self.framework_name_en = "Uncertainty-Driven Mean Teacher Framework"
        self.innovation1_name_cn = "UC-EDL"
        self.innovation1_name_en = "Pseudo-Label EDL Supervision and Uncertainty Calibration"
        self.innovation2_name_cn = "UA-EMA"
        self.innovation2_name_en = "Uncertainty-Aware Exponential Moving Average"

        # ========= Data =========
        self.data_root = "../../WXSOD_data"
        self.train_split = "train_sys"
        self.test_splits = ["test_sys", "test_real"]
        self.labeled_ratio = 0.1
        self.split_seed = 42
        self.save_split_list = True

        # ========= Runtime =========
        self.image_size = 352
        self.resize_min_scale = 0.8
        self.resize_max_scale = 1.0
        self.batch_size = 8
        self.num_workers = 4
        self.pin_memory = True
        self.persistent_workers = True
        self.prefetch_factor = 2
        self.use_amp = False
        self.device = None

        # ========= Model =========
        self.encoder_name = "pvt_v2_b2"
        self.encoder_pretrained = True
        self.encoder_pretrained_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "pvtv2b2.pth",
        )
        self.decoder_channels = 256

        # ========= Optimization =========
        self.max_epochs = 80
        self.steps_per_epoch = None
        self.lr = 1e-4
        self.min_lr = 1e-6
        self.weight_decay = 1e-4
        self.grad_clip = 1.0
        self.supervised_only_epochs = 15
        self.unsup_rampup_epochs = 20


        self.unsup_weight_max = 1.0
        self.edl_lambda_kl = 0.1
        self.pseudo_label_threshold = 0.5
        self.dual_ucc_conf_threshold = 0.01  


        self.quality_ema_momentum = 0.9
        self.quality_gamma = 2.0
        self.teacher_ema_supervised = 0.999
        self.teacher_ema_min = 0.9900
        self.teacher_ema_max = 0.9999

        # ========= Normalization =========
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

        # ========= Output =========
        self.exp_root = "experiments"
        self.exp_name = "ssod"
        self.save_every = 10
        self.save_pred_dir = "predictions"

    def to_dict(self):
        return dict(self.__dict__)

    def framework_summary(self):
        return {
            "framework_cn": self.framework_name_cn,
            "framework_en": self.framework_name_en,
            "innovation1_cn": self.innovation1_name_cn,
            "innovation1_en": self.innovation1_name_en,
            "innovation2_cn": self.innovation2_name_cn,
            "innovation2_en": self.innovation2_name_en,
        }

    @classmethod
    def from_dict(cls, payload):
        cfg = cls()
        for key, value in payload.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg
