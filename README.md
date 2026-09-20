# rice-leaf-disease-hybrid-tribackbone

# Rice Leaf Disease Classification — Hybrid Tri-Backbone

A hybrid deep learning model combining EfficientNetB0 + ConvNeXt-Tiny + DaViT-Tiny 
with adaptive attention fusion for rice leaf disease classification (8 classes, 97.66% accuracy).

## Dataset
Download the dataset (2560 images, 8 classes) from:
👉 [Your Google Drive / Kaggle link here]

After downloading, place it in your Google Drive and update the `INPUT_DIR` path 
in the notebook accordingly.

## How to Run
1. Open `rice_disease_classification.ipynb` in Google Colab 
   (click the "Open in Colab" badge below, or upload manually).
2. Mount your Google Drive.
3. Update the dataset path (`INPUT_DIR`) to match where you placed the dataset.
4. Run all cells sequentially.

## Requirements
See `requirements.txt` — installed automatically in the first notebook cell.

## Results
| Model | Accuracy |
|---|---|
| EfficientNetB0 | 95.83% |
| ConvNeXt-Tiny | 96.09% |
| DaViT-Tiny | 97.14% |
| **Hybrid (Proposed)** | **97.66%** |

## Authors
Syed Amin Hussain, Al-Arafat Hossain Shabbir  
Supervisor: Samia Rahman Rima  
Metropolitan University, Sylhet
