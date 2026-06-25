# Uncertainty-Driven Mean Teacher Framework

This repository provides the PyTorch implementation of **UDMT**, an
uncertainty-driven mean teacher framework for semi-supervised
condition-constrained salient object detection.

## Environment

The code was developed for Python 3.8+ and PyTorch. A CUDA-enabled GPU is
recommended for training.

```bash
git clone https://github.com/TurnHug/UDMT-main.git
cd UDMT-main

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The default backbone is `pvt_v2_b2`. Please place the local pretrained PVTv2-B2
weights at:

```text
UDMT-main/pvtv2b2.pth
```

Alternatively, pass a custom path with `--encoder_pretrained_path`.

## Dataset Layout

By default, `config.py` expects WXSOD-style data under `../../WXSOD_data`.
You can also pass `--data_root` explicitly. Each split should contain an image
folder and a mask folder. The loader supports common folder names such as
`input`, `images`, `image`, or `imgs` for images, and `gt`, `GT`, `mask`, or
`masks` for masks.

Example layout:

```text
WXSOD_data/
  train_sys/
    input/
    gt/
  test_sys/
    input/
    gt/
  test_real/
    input/
    gt/
```

## Training

The following command reproduces the default 10% labeled WXSOD setting used in
the paper:

```bash
python train.py \
  --data_root /path/to/WXSOD_data \
  --train_split train_sys \
  --test_splits '["test_sys","test_real"]' \
  --labeled_ratio 0.1 \
  --split_seed 42 \
  --max_epochs 80 \
  --batch_size 8 \
  --encoder_pretrained_path /path/to/pvtv2b2.pth
```

Training outputs are saved under `experiments/`. The code also writes the
labeled/unlabeled split file to the experiment directory when
`save_split_list=True`.

## Evaluation

After training, evaluate the latest experiment with:

```bash
python eval.py
```

Or evaluate a specific experiment/checkpoint:

```bash
python eval.py \
  --experiments-dir experiments/ssod_YYYYMMDD_HHMMSS \
  --checkpoint experiments/ssod_YYYYMMDD_HHMMSS/checkpoints/teacher_latest.pth
```

The evaluation script reports MAE, mean F-measure, mean E-measure, and
S-measure, and saves the results to:

```text
experiments/<experiment_name>/logs/eval_results.json
```

## Inference

To save saliency prediction maps for a test split:

```bash
python test.py \
  --experiments-dir experiments/ssod_YYYYMMDD_HHMMSS \
  --split test_real \
  --save-dir predictions/test_real
```

For an external dataset folder:

```bash
python test.py \
  --checkpoint /path/to/teacher_latest.pth \
  --data-path /path/to/dataset_split \
  --save-dir predictions/custom_split
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for
details.
