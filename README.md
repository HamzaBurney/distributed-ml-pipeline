# Distributed and Parallel ML Pipeline for Binary Classification
This project showcases the development of an optimized machine learning pipeline for binary classification using structured data. It compares the performance of traditional and accelerated approaches, including multi-core CPU execution, distributed systems (Dask + Coiled), and GPU acceleration (TensorFlow). The main models used are XGBoost and Neural Networks.

## 🎯 Objectives

- Maximize model accuracy  
- Minimize training time (by at least 70%)  
- Leverage GPU acceleration, multi-threading, and distributed systems  

---

## 🔄 Preprocessing Pipeline

- **Duplicate Removal**  
- **Missing Value Imputation**  
  - Categorical: Mode  
  - Numerical: Mean  
- **Outlier Detection & Removal**  
- **One-Hot Encoding** for categorical variables  
- **StandardScaler Normalization**

---

## 🧠 Modeling Approaches

### 🔸 Model 1: XGBoost Classifier
- **Baseline**: Single-core training with `XGBClassifier`
- **Optimized**: Distributed training using `dask_ml.xgboost` with Coiled clusters

### 🔹 Model 2: Neural Network (TensorFlow)
- **Baseline**: Sequential model on single-core CPU
- **Multi-Core**: Thread-parallel training using TensorFlow’s CPU optimizations
- **GPU**: Final model trained on NVIDIA GPU for speed and improved convergence

---

## 📊 Evaluation Metrics

- **Accuracy**
- **F1 Score**
- **Confusion Matrix**
- **Processing Time (Seconds)**

---

## 📈 Benchmark Results

| Model Variant                  | Avg. Processing Time Reduction |
|-------------------------------|-------------------------------|
| XGBoost (CPU → Dask)          | 91%                          |
| Neural Network (1-Core → GPU) | 70%                          |
| Neural Network (1 → Multi)    | 39%                          |
| Neural Network (Multi → GPU)  | 45%                          |

> All model variants achieved over **60% accuracy**.

---

## ⚖️ Comparative Analysis

### XGBoost
- ✅ Great for structured/tabular data  
- ✅ Scales efficiently with Dask  
- ❌ Less expressive for complex patterns

### Neural Network
- ✅ Highly expressive with GPU acceleration  
- ❌ Computationally expensive on CPU  
- ❌ Requires hyperparameter tuning

---

## 🧪 Experimental Setup

### Hardware
- CPU: Intel-based
- GPU: NVIDIA P100 (Kaggle)

### Software
- Python 3.13  
- Libraries:
  - `scikit-learn`
  - `xgboost`
  - `tensorflow`
  - `dask`, `coiled`
  - `pandas`, `numpy`

### Environments
- Local Jupyter Notebook  
- Kaggle Notebook  
- Coiled Cloud Cluster (for Dask)

---

## 🚀 Future Work

- Hyperparameter optimization via Ray Tune  
- Deploy as an API using FastAPI  
- Extend to multi-class classification

---

## 📜 License

This project is for academic purposes. Contact the authors for reuse.

---

