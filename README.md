# Geologically Constrained Machine Learning for Hydraulic Flow Unit Identification

**Paper:** Geologically Constrained Machine Learning for Hydraulic Flow Unit Identification in Tight Sandstone Reservoirs: A Case Study from the Shaximiao Formation, Zitong Area  
**Authors:** Xiaolong He¹, Bing Zhang¹²*, Chuan Xu¹, Kai Yang¹, Yifan He¹, Zhuo Li¹, Dongxing Wang¹, Ken Cheng¹, Yangsen Gao¹

---

## 1. Project Overview
This repository contains the code and data supporting the identification of Hydraulic Flow Units (HFUs) in tight sandstone reservoirs using machine learning techniques constrained by geological information.  

Key contributions include:

- **HFU classification** based on petrophysical and sedimentological data.  
- **Extraction of GR curve morphology parameters** for characterizing reservoir heterogeneity.  
- **XGBoost model training** for predictive classification of HFUs.  

This work is based on the Shaximiao Formation, Zitong Area.

---

## 2. Repository Structure

├── data/ # Raw and processed datasets used in the study
├── scripts/ # Python scripts for data processing and modeling
├── README.md # Project description and usage instructions



---

## 3. Data Description
- `data/XGBoost_sample.txt` — Tab-separated file containing features and labels for HFU prediction.  
- Columns include GR curve parameters and other petrophysical features.  
- The `标签` column is the HFU class label.  

> ⚠️ **Note:** Please check the data file for missing values (NA, /, etc.) before processing.

---

## 4. Scripts

| Script Name              | Description |
|--------------------------|-------------|
| `scripts/feature_processing.py` | Feature extraction from GR curves and other petrophysical properties. |
| `scripts/XGBoost.py`     | Train XGBoost model for HFU prediction, includes SMOTETomek balancing and evaluation. |
| `scripts/visualization.py` | Visualization of GR curves, HFU distributions, and model performance. |

---

## 5. Installation & Requirements

```bash
# Clone the repository
git clone https://github.com/<username>/<repo>.git
cd <repo>

# Create environment (optional but recommended)
conda create -n hfu_ml python=3.10
conda activate hfu_ml

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn


## 6. Usage Example

# Example: train HFU model
from scripts.XGBoost import train_model

# Load data
X, y = train_model.load_data("data/XGBoost_sample.txt")

# Train and evaluate model
model, results = train_model.train(X, y)

# Plot model results
train_model.plot_results(results)

## 7. References

He, X., Zhang, B., Xu, C., et al. (2025). Geologically Constrained Machine Learning for Hydraulic Flow Unit Identification in Tight Sandstone Reservoirs: A Case Study from the Shaximiao Formation, Zitong Area. Journal Name.

