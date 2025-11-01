Hydraulic Flow Unit Identification
This repository contains the implementation code for the paper "Geologically Constrained Machine Learning for Hydraulic Flow Unit Identification in Tight Sandstone Reservoirs".

Overview
HFU Clustering: Hydraulic Flow Unit classification based on FZI

GR Shape Analysis: Extract morphological features from Gamma Ray curves

XGBoost Model: Machine learning model for HFU prediction

Project Structure
text
├── data/                    # Data files
├── scripts/                 # Code scripts
│   ├── hfu_clustering.py    # HFU classification
│   ├── gr_feature_extraction.py  # GR shape feature extraction
│   └── xgboost_model.py     # XGBoost training and prediction
└── README.md
Quick Start
Install dependencies:

bash
pip install pandas numpy scikit-learn xgboost matplotlib
Run the pipeline:

bash
python scripts/hfu_clustering.py
python scripts/gr_feature_extraction.py  
python scripts/xgboost_model.py
Data Format
Input data should include these well logs:

DEPTH - Depth

GR - Gamma Ray

PHIT - Porosity

PERM - Permeability

Citation
If you use this code, please cite our paper.
