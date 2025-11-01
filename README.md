# Geologically Constrained Machine Learning for Hydraulic Flow Unit Identification

**Paper:** Geologically Constrained Machine Learning for Hydraulic Flow Unit Identification in Tight Sandstone Reservoirs: A Case Study from the Shaximiao Formation, Zitong Area  
**Authors:** Xiaolong He¹, Bing Zhang¹²*, Chuan Xu¹, Kai Yang¹, Yifan He¹, Zhuo Li¹, Dongxing Wang¹, Ken Cheng¹, Yangsen Gao¹

¹Earth Exploration and Information Technology Key Laboratory of Ministry of Education, Chengdu University of Technology, Chengdu, Sichuan, China

²State Key Laboratory of Oil and Gas Reservoir Geology and Exploitation, Chengdu University of Technology, Chengdu, Sichuan, China

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

This repository contains the following data files:

| File Name                     | Description |
|-------------------------------|-------------|
| `Well stratification.txt`     | Well stratification information for each well. |
| `Single-well GR.txt`          | Single-well GR curve data. |
| `Morphological parameters.csv`| Morphological parameters of GR curves calculated per layer. |
| `LOG(FZI).txt`                | Logarithmic values of Hydraulic Flow Units (HFU). |
| `XGBoost_sample.txt`          | Dataset used for model training and testing (features + HFU labels). |

> ⚠️ **Note:** Check the data files for missing values (NA, /, etc.) before processing.

---

## 4. Scripts

This repository contains the following Python scripts:

| Script Name                       | Description |
|----------------------------------|-------------|
| `Morphological parameters.py`     | Computes morphological parameters of GR curves per layer. |
| `HFU Division.py`                 | HFU classification based on petrophysical and GR data. |
| `Hyperparameter Optimization.py`  | Hyperparameter optimization for XGBoost model with geological constraints. |
| `XGBoost.py`                      | XGBoost model training and evaluation with geological constraints. |
| `Hyperparameter Optimization_2.py`| Hyperparameter optimization for XGBoost model **without** geological constraints. |
| `XGBoost_2.py`                     | XGBoost model training and evaluation **without** geological constraints. |


---
