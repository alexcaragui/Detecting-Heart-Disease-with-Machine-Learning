# Detecting Heart Disease with Machine Learning

A binary classification neural network that predicts the presence of heart disease in patients based on clinical data from the Cleveland Heart Disease dataset.

---

## Overview

Heart disease is one of the leading causes of death worldwide. This project applies machine learning techniques to assist in early detection by analyzing patient medical data and predicting whether heart disease is present.

---

## Dataset

- **Source:** Cleveland Heart Disease Dataset
- **Instances:** 302 patients (296 after cleaning)
- **Features:** 13 clinical attributes including age, sex, chest pain type, resting blood pressure, cholesterol, and more
- **Target:** Binary — 0 (no heart disease) / 1 (heart disease present)

---

## Project Pipeline

1. **Data Loading & Exploration** — summary statistics, data types, initial inspection
2. **Data Cleaning** — replaced `?` values with `NaN`, dropped rows with missing values
3. **Class Balancing** — used `RandomOverSampler` to balance classes (159 vs 137 → 159 vs 159)
4. **Feature Scaling** — applied `MinMaxScaler` to normalize all features to [0, 1]
5. **Correlation Analysis** — Pearson correlation heatmap to identify feature relationships
6. **Model Building** — Sequential neural network with Dense and Dropout layers
7. **Model Evaluation** — Accuracy, Confusion Matrix, ROC/AUC, Precision-Recall, F1 Score

---

## Model Architecture

```
Input (13 features)
        ↓
Dense (15 neurons, ReLU)
        ↓
Dropout (0.2)
        ↓
Dense (15 neurons, ReLU)
        ↓
Dropout (0.4)
        ↓
Dense (1 neuron, Sigmoid)
        ↓
Output (0 or 1)
```

- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Epochs:** 800

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | ~89% |
| F1 Score | 0.88 |
| AUC | ~0.93 |

**Confusion Matrix:**
```
[[31  4]
 [ 3 26]]
```
- 31 True Negatives, 26 True Positives
- 4 False Positives, 3 False Negatives

---

## Technologies

| Category | Tools |
|----------|-------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Machine Learning | scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Class Balancing | imbalanced-learn |

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/alexcaragui/Detecting-Heart-Disease-with-Machine-Learning.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open the notebook:
```bash
jupyter lab
```

4. Run `main.ipynb` cell by cell

---

## Limitations & Future Improvements

- Dataset is relatively small (~300 instances) — results may vary
- `EarlyStopping` could be added to prevent overfitting
- Model could be compared against other classifiers (Random Forest, SVM, XGBoost)
- SMOTE could replace RandomOverSampler for more robust synthetic data generation
