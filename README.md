
Overview
HybridXAI is a robust, leakage-free, and explainable deep learning pipeline for multi-class skin lesion classification. The architecture fuses the global contextual power of the Swin Transformer with the fine-grained local texture extraction of ConvNeXtV2 using an asymmetric cross-attention mechanism.

Designed for the HAM10000 dataset, this model addresses critical gaps in automated dermoscopy:

Patient-level data leakage: Employs a strict GroupShuffleSplit to ensure zero lesion-ID overlap between train/val/test splits.

Representational bottlenecks: Fuses local and global features efficiently, yielding superior accuracy compared to single-backbone or naive concatenation methods.

Clinical transparency deficit: Integrates a three-tier multi-granularity Explainable AI (XAI) suite (Grad-CAM, LIME, GradientSHAP) for clinician-interpretable predictions.

Our model achieves 99.94% Test Accuracy and 1.000 AUC-ROC on the held-out test partition.

Architecture
The dual-stream architecture consists of ~178.7M parameters:

Backbone A: Swin Transformer Base (captures global lesion structure/geometry).

Backbone B: ConvNeXtV2-Base (captures high-frequency local textures like vascular patterns).

Fusion Module: Asymmetric Cross-Attention (Swin features serve as Queries; ConvNeXt features serve as Keys/Values).

Classification Head: Two-layer fully connected head predicting 7 skin lesion classes.


# HybridXAI
The notebook executes the following 5 phases
The notebook executes the following 5 phases:

Leakage-Free Data Splitting:  Partitions the data using lesion_id.

Class Balancing & Augmentation: Applies focal loss, WeightedRandomSampler, CutMix, and RandAugment.

Architecture Instantiation: Loads pre-trained Swin and ConvNeXtV2 models and builds the Cross-Attention fusion module.

Centralized Training: Trains using AdamW, Cosine Annealing Warm Restarts, Automatic Mixed Precision (AMP), and Exponential Moving Average (EMA).

Evaluation & XAI: Runs three-view Test-Time Augmentation (TTA) inference and generates multi-granularity explainability maps.

Explainable AI (XAI) Suite
The pipeline includes a three-tier XAI module to ensure clinical transparency:

Macroscopic (Grad-CAM): Operates on the ConvNeXtV2 last convolutional layer to highlight broad lesion-level focus.

Superpixel (LIME): Constructs locally faithful linear surrogate models on SLIC superpixels.

Pixel-Level (GradientSHAP): Computes pixel-level attribution volumes to identify fine-grained supportive or contradictory features.

HybridXAI/
│
├── data/                   # Directory for the HAM10000 dataset & metadata
├── notebooks/              
│   └── HybridXAI.ipynb     # Main Jupyter notebook containing the full pipeline
├── src/                    # Source code (if exported from notebook)
│   ├── data_loader.py      # Leakage-free splitting and augmentation
│   ├── model.py            # HybridXAI architecture definition
│   ├── train.py            # Centralized training with AMP and EMA
│   └── xai_utils.py        # Grad-CAM, LIME, and GradientSHAP implementation
├── weights/                # Directory to save the best EMA checkpoints
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
