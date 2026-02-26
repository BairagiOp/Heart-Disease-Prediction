# 🫀 Heart Disease Prediction — Streamlit App

A local Streamlit web app for Heart Disease prediction using a Decision Tree Classifier.

---

## 📁 Required Files

Make sure ALL three files are in the **same folder**:

```
your_folder/
├── app.py                          ← Streamlit app
├── heart_disease_model.pkl         ← Trained model (from Colab)
└── requirements.txt                ← Dependencies
```

---

## 🚀 Setup & Run

### Step 1 — Get the pickle file from Colab
Add this as the last cell in your Colab notebook and run it:
```python
import pickle
with open("heart_disease_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
from google.colab import files
files.download("heart_disease_model.pkl")
```
Move the downloaded file into the same folder as `app.py`.

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the app
```bash
streamlit run app.py
```

The app opens automatically at → **https://heart-disease-prediction-xxonyzfttsvrl5swgqfomu.streamlit.app/**

---

## ✨ App Features

| Tab | What you get |
|-----|-------------|
| 📜 Decision Path | Step-by-step trace of exactly how the tree reached its prediction |
| 🌳 Tree Visualization | Full coloured decision tree (adjustable depth) + raw text rules |
| 📊 Feature Importance | Bar chart of all 13 features ranked by Gini importance |
| 🧾 Patient Summary | All input values + automatic risk flag detection |

---

## 🎛️ Input Features (Sidebar)

| Feature | Range / Options |
|---------|----------------|
| Age | 29 – 77 years |
| Sex | Male / Female |
| Chest Pain Type | Typical · Atypical · Non-Anginal · Asymptomatic |
| Resting BP | 94 – 200 mmHg |
| Cholesterol | 126 – 564 mg/dl |
| Fasting Blood Sugar | Yes / No |
| Resting ECG | Normal · ST-T Abnormality · LVH |
| Max Heart Rate | 71 – 202 bpm |
| Exercise Angina | Yes / No |
| ST Depression | 0.0 – 6.2 |
| ST Slope | Downsloping · Flat · Upsloping |
| Major Vessels | 0 – 3 |
| Thalassemia | Normal · Fixed Defect · Reversible Defect |

---

> ⚠️ **Disclaimer**: This app is for educational and research purposes only.
> It is not intended for clinical diagnosis or medical decision-making.
