# 🧬 ML Portfolio — SVM Classification & Regression

**Production-ready end-to-end Machine Learning system**
From raw data → modeling → evaluation → deployed web application

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 🚀 Overview

This project demonstrates a complete machine learning workflow across two real-world problems:

* 🔬 **Binary Classification** — Breast Cancer Diagnosis (SVC)
* 🏘️ **Regression** — Housing Price Prediction (SVR)

It goes beyond notebooks by delivering a **fully interactive Streamlit application** for real-time predictions, insights, and model exploration.

---

## 💭 Why This Project?

Most machine learning projects stop at model accuracy.
This project focuses on building a **complete system**:

* Reproducible pipelines
* Clean feature engineering
* Reliable model evaluation
* Interactive user interface

Designed to simulate real-world ML deployment scenarios.

---

## 📁 Project Structure

```bash
ml-svm-portfolio/
├── 01_SVC_BreastCancer.ipynb
├── 02_SVR_Housing.ipynb
├── app.py
├── breast-cancer.csv
├── BostonHousing.csv
└── models/
    ├── svc_breast_cancer.pkl
    └── svc_metadata.pkl
```

---

## ✨ Key Features

### 📊 Full EDA

* Statistical summaries, distributions, outlier detection
* Correlation heatmaps & visual exploration

### ⚙️ Feature Engineering

* Custom features:
  * `TumorSize`
  * `CompactnessLevel`
  * `PriceCategory`
  * `NearRiver`
  * `HighCrime`

### 🔄 Robust Pipelines

* `ColumnTransformer` → Imputation → Scaling → Model
* Fully reproducible preprocessing workflow

### 🎯 Model Optimization

* `GridSearchCV` with cross-validation
* Hyperparameter tuning for:
  * SVC (C, gamma, kernel)
  * SVR (C, epsilon, gamma)

### 📈 Model Explainability

* Permutation Feature Importance (model-agnostic)

### 🖥️ Interactive Web App

* Manual prediction (sliders & inputs)
* Batch CSV upload with downloadable results
* Radar chart vs dataset averages
* Model insights dashboard

### 💾 Model Persistence

* Auto-save & auto-load using `joblib` and `pickle`

---

## 🛠️ Tech Stack

| Category          | Technologies                              |
| ----------------- | ----------------------------------------- |
| **Core ML**       | Python 3.10+, scikit-learn, pandas, NumPy |
| **Modeling**      | SVC (RBF Kernel), SVR, GridSearchCV       |
| **Engineering**   | Pipeline, ColumnTransformer               |
| **Visualization** | Streamlit, Matplotlib, Seaborn            |
| **Utilities**     | scipy, joblib, pickle                     |

---

## 📊 Model Performance

### 🔬 Breast Cancer Classification (SVC)

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 97.37% |
| ROC-AUC   | 0.9947 |
| Precision | 0.97   |
| Recall    | 0.96   |

> ✅ Near-perfect class separation

---

### 🏘️ Housing Price Regression (SVR)

| Metric   | Score |
| -------- | ----- |
| R² Score | 0.838 |
| RMSE     | 3.448 |
| MAE      | 2.156 |

> ✅ Strong predictive performance on structured data

---

## 🖥️ Application Preview

| Page                 | Features                           |
| -------------------- | ---------------------------------- |
| 🏠 Home              | Metrics dashboard & overview       |
| 🔍 Manual Prediction | Real-time prediction + radar chart |
| 📁 Batch Upload      | CSV upload & bulk predictions      |
| 📊 Model Insights    | Feature importance & analysis      |

> 📸 *Add screenshots or GIF here for maximum impact*

---

## 🎬 Live Demo

🚀 Coming soon (Streamlit Cloud deployment)

---

## ▶️ How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/ml-svm-portfolio.git
cd ml-svm-portfolio
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash
streamlit run app.py
```

---

## ⚙️ Notes

* The app automatically trains models if `.pkl` files are missing
* All model artifacts are saved inside the `models/` directory
* Ensure datasets are placed in the root directory:
  * `breast-cancer.csv`
  * `BostonHousing.csv`

---

## 🔮 Future Improvements

* 🔍 SHAP explainability (local & global)
* ⚡ Benchmark against XGBoost / LightGBM
* 📊 Prediction uncertainty (conformal prediction)
* ☁️ Deploy to Streamlit Cloud
* 🧪 Add unit testing (pytest)
* 📦 Dockerize the application

---

## 🤝 Contributing

Contributions, ideas, and improvements are welcome.
Feel free to fork the repository and open a pull request.

---

## ⭐ Support

If you found this project useful, consider giving it a star ⭐

---

## 📫 Contact

* LinkedIn: *(https://www.linkedin.com/in/mohamed-marof/)*
* GitHub: *(https://github.com/mohmedmarof2551997-sudo)*

---

## ⚡ Final Note

Accuracy is just the beginning.
Real impact comes from building systems people can actually use.
