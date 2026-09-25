# 🌱 Crop Disease Detection

A deep learning-based web application that identifies potato leaf diseases from images using a Convolutional Neural Network (CNN).

## 📌 Project Overview

This project detects three potato leaf conditions:

* 🥔 Potato Early Blight
* 🥔 Potato Late Blight
* 🥔 Healthy Potato Leaf

The user uploads an image of a potato leaf, and the trained deep learning model predicts the disease condition.

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* CNN
* FastAPI
* React.js
* Vite
* JavaScript
* HTML
* CSS
* OpenCV

## 📂 Project Structure

```text
crop-disease-detection/
│
├── backend/
│   ├── model/
│   ├── class_names.json
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── model-training/
│
├── dataset/
│
└── .gitignore
```

## ⚙️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Akashjaiswal02/crop-disease-detection.git
cd crop-disease-detection
```

### 2. Run Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend will run at:

```text
http://127.0.0.1:8000
```

### 3. Run Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend will run at:

```text
http://localhost:5173
```

## 🔍 How It Works

```text
Leaf Image
    ↓
Image Preprocessing
    ↓
CNN Deep Learning Model
    ↓
Disease Classification
    ↓
Prediction Result
```

## 🎯 Objective

The objective of this project is to use deep learning and computer vision techniques to help identify potato leaf diseases from images.

## 👨‍💻 Author

**Akash Jaiswal**

B.Tech CSE
NIET, Greater Noida

GitHub: https://github.com/Akashjaiswal02
