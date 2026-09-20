# Rice Leaf Disease Classification — Hybrid Tri-Backbone Deep Learning Model

## 📌 Project Overview

Rice is one of the most important staple crops in Bangladesh and across South and Southeast Asia, 
but rice leaf diseases significantly reduce yield and threaten food security every year. Early and 
accurate detection of these diseases is critical for farmers to take timely action, yet manual 
diagnosis by visual inspection is slow, subjective, and requires expert knowledge that is often 
unavailable in rural farming communities.

This project proposes a **Hybrid Tri-Backbone Deep Learning Model** that combines three 
state-of-the-art architectures — **EfficientNetB0**, **ConvNeXt-Tiny**, and **DaViT-Tiny** — through 
an **Adaptive Attention Fusion mechanism**, to accurately classify 8 different rice leaf disease 
categories from leaf images. The proposed model achieves **97.66% accuracy**, outperforming each 
individual backbone architecture used alone.

## 🎯 Objectives / What We Are Trying to Achieve

1. **Build a highly accurate rice leaf disease classifier** by combining the complementary 
   strengths of a CNN (EfficientNetB0), a modern ConvNet (ConvNeXt-Tiny), and a Vision Transformer 
   (DaViT-Tiny) — rather than relying on a single architecture.
2. **Design an Adaptive Fusion mechanism** that automatically learns how much "attention" or 
   importance to give each backbone for a given input, instead of using fixed/manual weighting.
3. **Prove the model generalizes to real-world, unseen data** — not just the training dataset — 
   through rigorous cross-dataset/external validation.
4. **Go beyond raw accuracy** by evaluating the model from multiple angles that matter for 
   real-world deployment: interpretability, efficiency, robustness, and statistical significance.
5. **Provide a practical, deployable solution** that could eventually support farmers in Bangladesh 
   and similar rice-growing regions with fast, low-cost, image-based disease diagnosis.

## 🏗️ Methodology

### Model Architecture
- **Three backbones** extract complementary features from each input leaf image:
  - **EfficientNetB0** — efficient convolutional feature extraction
  - **ConvNeXt-Tiny** — modernized CNN with strong hierarchical features
  - **DaViT-Tiny** — dual attention vision transformer, capturing global context
- **Adaptive Tri-Fusion module** — a learned attention gate that dynamically weighs each 
  backbone's contribution per image/class, rather than simple averaging or concatenation.
- **Auxiliary classification heads** (per backbone) support training stability and enable 
  statistical comparison (see Novelty #5 below).
- **EMA (Exponential Moving Average)** weight smoothing and **TTA (Test-Time Augmentation)** 
  are used to further stabilize and improve inference accuracy.

### Dataset
- 8 disease classes: Bacterial Leaf Blight, Brown Spot, Healthy Rice Leaf, Leaf Blast, 
  Leaf scald, Narrow Brown Leaf Spot, Rice Hispa, Sheath Blight
- 2,560 images, balanced via augmentation (650 images/class)
- Split: 70% train / 15% validation / 15% test

## 🌟 Novelty & Key Contributions

This work goes beyond standard classification benchmarking with **6 additional experiments**:

| # | Experiment | Purpose |
|---|---|---|
| 1 | **Cross-Dataset / External Validation** | Verified on a completely independent, leak-free external dataset (4,738 images, duplicate-checked via MD5 hashing) — achieved **96.10% accuracy**, only a 1.56% drop from validation accuracy, proving genuine generalization rather than memorization. |
| 2 | **Backbone Attention-Weight Analysis** | Reveals *which* backbone the fusion mechanism relies on for each disease class — an interpretability contribution showing the fusion is genuinely adaptive, not just a black box. |
| 3 | **Efficiency / Inference-Time Benchmarking** | Measures parameter count and inference speed of the Hybrid model vs. each baseline, quantifying the accuracy-vs-efficiency trade-off for real-world deployment feasibility. |
| 4 | **Robustness Testing** | Evaluates performance under realistic image degradations (blur, low brightness, sensor noise) — accuracy stayed above 95% in all cases, confirming resilience to non-ideal field conditions (e.g., farmer smartphone photos). |
| 5 | **McNemar's Statistical Significance Test** | Statistically proves (p < 0.0001) that the Hybrid fusion's improvement over a single backbone is real and not due to random chance. |
| 6 | **t-SNE Feature Embedding Visualization** | Visually confirms the model learns well-separated, discriminative feature clusters per disease class. |

## 📊 Results

### Baseline vs. Hybrid Model Comparison
| Model | Accuracy |
|---|---|
| Swin Transformer | 94.53% |
| EfficientNetB0 | 95.83% |
| ConvNeXt-Tiny | 96.09% |
| DaViT-Tiny | 97.14% |
| **Hybrid Tri-Backbone (Proposed)** | **97.66%** |

### Cross-Dataset External Validation
| Dataset | Images | Accuracy |
|---|---|---|
| Original (Validation Set) | 384 | 97.66% |
| External (Independent, Leak-Free) | 4,738 | 96.10% |

## 📂 Dataset Access

Download the training dataset (2,560 images, 8 classes) here:
👉 [Google Drive Dataset Link](https://drive.google.com/drive/folders/1FLfSiRuPap455ry_7XUeXPoLIB4-7ofV?usp=sharing)

After downloading, upload it to your own Google Drive and update the `INPUT_DIR` variable 
in the notebook to match its location.

## 🚀 How to Run

1. Open `rice_disease_classification.ipynb` in **Google Colab**.
2. Mount your Google Drive when prompted.
3. Download the dataset (link above) and place it in your Drive.
4. Update the `INPUT_DIR` path in the notebook to point to your dataset location.
5. Run all cells sequentially (**Runtime → Run all**), or run cell-by-cell if you 
   only want to re-run specific sections (e.g., the novelty experiments at the end).

> ⚠️ Note: Full training from scratch (all 4 baselines + hybrid model) can take several hours 
> on Colab's free GPU tier. Trained model weights (`.pth` files) are saved to Google Drive 
> automatically, so training does not need to be repeated once complete.

## 🛠️ Requirements

See `requirements.txt`. Key libraries: PyTorch, torchvision, timm, Albumentations, 
pytorch-grad-cam, scikit-learn, statsmodels.

## 🔍 Explainability

Grad-CAM visualizations are included to highlight which regions of the leaf the model 
focuses on when making predictions, supporting trust and interpretability for practical use.

## ⚠️ Limitations & Future Work

- The "Narrow Brown Leaf Spot" class shows comparatively lower performance on external data, 
  likely due to distributional shift in imaging conditions — a direction for future improvement.
- Future work includes deploying the model as a lightweight mobile application for real-time 
  field diagnosis, and expanding validation with multi-source Bangladeshi field datasets.

## 👥 Authors

- Syed Amin Hussain
- Al-Arafat Hossain Shabbir

**Supervisor:** Samia Rahman Rima  
**Institution:** Metropolitan University, Sylhet
