# diabetes-risk-predictor
# 🩺 Diabetes Risk Prediction Web App

This is a beginner-friendly Machine Learning project that predicts the risk of diabetes using medical input data. The project is built with Flask, Scikit-learn, and HTML/CSS and demonstrates how to deploy an ML model as a web application.

![App Screenshot](static/logo.png)

---

## 🚀 Live Demo

*(Optional: Add GitHub Pages or Render/Streamlit link here if deployed)*

---

## 📌 Features

- Predict diabetes risk based on 8 medical parameters
- Logistic Regression Model
- Clean, responsive HTML/CSS interface
- Dynamic graph generation using Matplotlib
- Flask backend with form input and result display

---

## 🧠 Tech Stack

- Python
- Flask
- Pandas, NumPy, Scikit-learn
- Matplotlib
- HTML/CSS (Vanilla)
- Jinja2 templating (Flask)

---

## 📁 Folder Structure

diabetes-risk-predictor/
│
├── app.py # Main Flask app
├── diabetes_model.pkl # Trained ML model
├── scaler.pkl # Scaler used for feature scaling
│
├── /templates/
│ └── index.html # Web interface
│
├── /static/
│ ├── styles.css # CSS styling
│ ├── logo.png # Header image
│ └── graph.png # Generated graph (auto)
│
├── README.md # Project documentation
└── requirements.txt # Python dependencies