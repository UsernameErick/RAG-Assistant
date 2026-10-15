# Pneumonia Detection — Chest X-Ray Classification (PyTorch)

Binary classification of chest X-ray images: **NORMAL** vs **PNEUMONIA**.
Built as a hands-on introduction to PyTorch through a production-style case (transfer learning, proper train/val/test evaluation, confusion matrix).

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle, Paul Mooney)
- 5216 images in train, 624 in test
- The original val split (16 images) wasn't used — too small for a stable metric. Instead, 15% of train was carved out as val via `random_split` (seed=42)

## Approach

- **Backbone:** ResNet18, pretrained on ImageNet
- **Transfer learning:** most of the network frozen; `fc` layer replaced (2 classes instead of 1000) and trained, plus `layer4` unfrozen and fine-tuned
- **Augmentation** (train only): random horizontal flip, rotation up to 10°
- **Normalization:** standard ImageNet mean/std
- **Optimizer:** Adam with discriminative learning rates — `fc`: `lr=0.001`, `layer4`: `lr=0.0001` — plus `weight_decay=1e-4` (regularization)
- A simpler `fc`-only baseline (no `layer4` fine-tuning) was tried first and got slightly higher accuracy (85.7%) but missed more pneumonia cases (6 vs. 2) — the fine-tuned version was kept as final, since catching pneumonia matters more than a couple points of accuracy

## Results (test set, 624 images)

| Metric | NORMAL | PNEUMONIA |
|---|---|---|
| Precision | 0.99 | 0.80 |
| Recall | 0.60 | 0.99 |
| F1 | 0.74 | 0.89 |

**Accuracy: 84.6%**

The model leans heavily toward catching pneumonia (recall 0.99, only 2 missed cases out of 390) at the cost of more false alarms on healthy patients (94 out of 234 misclassified). That trade-off makes sense for a screening tool, but the model is meant as a decision-support aid — not a replacement for a doctor's judgment.

## Project structure

```
data_setup.py   # data loading, transforms, train/val/test split
model_setup.py  # ResNet18 + transfer learning setup
train.py        # training loop
evaluate.py     # test-set evaluation, confusion matrix
```

## Running it

```bash
pip install torch torchvision scikit-learn
python train.py       # trains the model, saves model.pth
python evaluate.py    # evaluates on the test set
```

## What I learned

- Transfer learning: freezing layers vs. fine-tuning, and when each approach makes sense
- Discriminative learning rates for different parts of a network
- Why a large enough val split matters for a stable, trustworthy metric
- Confusion matrix and recall/precision as a more informative alternative to plain accuracy, especially in a medical context
- A model can look better on one metric (accuracy) and worse on the one that actually matters for the use case (missed positives) — the two don't always move together