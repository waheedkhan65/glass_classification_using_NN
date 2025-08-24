# 🔍 Glass Classification using Neural Network (PyTorch + Streamlit)

This project is an **end-to-end machine learning application** for classifying glass types based on their chemical composition. It uses a **custom Neural Network (PyTorch)** for model training and a **Streamlit web interface** for deployment, making it easy to interact with the model.

---

## 📌 Project Overview
- **Dataset:** UCI Glass Identification Dataset
- **Model:** Fully Connected Neural Network with 2 hidden layers (PyTorch)
- **Deployment:** Streamlit web app for real-time predictions
- **Key Features:**
  - End-to-end workflow (data preprocessing → training → deployment)
  - Standardization using `StandardScaler` (ensures consistent scaling during inference)
  - User-friendly Streamlit interface
  - Model and scaler persistence using `joblib` and `torch.save`

---

## 🚀 Tech Stack
- **Python**
- **PyTorch**
- **Streamlit**
- **Scikit-learn**
- **Pandas & NumPy**

---

## 🛠️ Project Structure

glass_classification_using_NN/
│
├── data/ # Dataset files
├── models/ # Trained model and scaler
├── src/ # Model architecture and utilities
│ ├── nn_model_architecture.py
│ ├── data_utils.py
│ └── train_model.py
├── app/ # Streamlit app
│ └── main.py
├── requirements.txt
└── README.md


---

## 📊 Dataset Details
- **Features:** RI, Na, Mg, Al, Si, K, Ca, Ba, Fe
- **Target:** Glass Type (7 categories)

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/waheedkhan65/glass_classification_using_NN.git
cd glass_classification_using_NN

pip install -r requirements.txt

streamlit run app/main.py
```

🎯 Key Learnings
  * Building and training Neural Networks in PyTorch
  * Preprocessing and standardization for consistent predictions
  * Saving and loading models & scalers for real-world use
  * Deploying ML models with Streamlit

📸 Demo
   <img width="1440" height="750" alt="stremmmmmmmmmmmmmmmm" src="https://github.com/user-attachments/assets/dda1b482-1d83-4b84-8fa4-dd9b481110cf" />

🔗 Connect
  GitHub: @waheedkhan65
  LinkedIn: https://www.linkedin.com/in/waheed-ur-rahman-54a514251/

