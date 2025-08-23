import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from src.nn_model_architecture import Model
import numpy as np
import joblib
import os


# Load Trained Model & Scaler
MODEL_PATH = "models/my_nn_model.pt"
SCALER_PATH = "models/scaler.pkl"

# Load scaler
scaler = joblib.load(SCALER_PATH)

# Initialize model architecture
model = Model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()


# Streamlit UI
st.set_page_config(page_title="Glass Classification", page_icon="🔍", layout="centered")
st.title("🔍 Glass Type Classification using Neural Network")

st.write("""
This app predicts the **type of glass** based on its chemical composition using a trained Neural Network model.
Provide the feature values below:
""")


# Feature Inputs
feature_names = [
    "RI: refractive index", "Na: Sodium", "Mg: Magnesium", "Al: Aluminum", "Si: Silicon", "K: Potassium", "Ca: Calcium", "Ba:  Barium", "Fe:  Iron"
]

input_data = []
cols = st.columns(3)
for i, feature in enumerate(feature_names):
    value = cols[i % 3].number_input(f"{feature}", min_value=0.0, max_value=20.0, value=1.0, step=0.01)
    input_data.append(value)


# Predict Button
if st.button("🔍 Predict Glass Type"):
    # Convert input to numpy and apply scaler
    X_input = np.array(input_data).reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    # Convert to tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        output = model(X_tensor)
        probs = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probs).item()
    
    # Class mapping
    glass_types = {
        0: "Building Windows (Float Processed)",
        1: "Building Windows (Non-Float Processed)",
        2: "Vehicle Windows (Float Processed)",
        3: "Vehicle Windows (Non-Float Processed)",
        4: "Containers",
        5: "Tableware",
        6: "Headlamps"
    }

    st.success(f"### ✅ Predicted Glass Type: **{glass_types[predicted_class]}**")
    st.write("#### Prediction Probabilities:")
    st.bar_chart(probs.numpy()[0])

st.write("---")
st.write("**📌 Model:** Neural Network (5 Hidden Layers, PyTorch)  |  **Dataset:** UCI Glass Dataset")
st.write("**🔗 Source Code:** [https://github.com/waheedkhan65/glass_classification_using_NN]")















# import streamlit as st
# import torch
# import torch.nn.functional as F
# import pandas as pd
# from src.nn_model_architecture import Model
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import os

# # ---------------------
# # Load Trained Model
# # ---------------------
# MODEL_PATH = "models/my_nn_model.pt"

# # Initialize model architecture
# model = Model()
# model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
# model.eval()

# # ---------------------
# # Streamlit UI
# # ---------------------
# st.set_page_config(page_title="Glass Classification", page_icon="🔍", layout="centered")
# st.title("🔍 Glass Type Classification using Neural Network")

# st.write("""
# This app predicts the **type of glass** based on its chemical composition using a trained Neural Network model.
# Provide the feature values below:
# """)

# # ---------------------
# # Feature Inputs
# # ---------------------
# feature_names = [
#     "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"
# ]

# input_data = []
# cols = st.columns(3)
# for i, feature in enumerate(feature_names):
#     value = cols[i % 3].number_input(f"{feature}", min_value=0.0, max_value=20.0, value=1.0, step=0.01)
#     input_data.append(value)

# # ---------------------
# # Predict Button
# # ---------------------
# if st.button("🔍 Predict Glass Type"):
#     # Preprocess: Standardize like training
#     scaler = StandardScaler()
#     # Fit with dummy (since we don't have training scaler saved, assume zero-mean std=1 approx)
#     # Real-world: Save the original scaler during training
#     X_input = np.array(input_data).reshape(1, -1)
#     X_scaled = scaler.fit_transform(X_input)

#     # Convert to tensor
#     X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

#     # Predict
#     with torch.no_grad():
#         output = model(X_tensor)
#         probs = F.softmax(output, dim=1)
#         predicted_class = torch.argmax(probs).item()
    
#     # Class mapping
#     glass_types = {
#         0: "Building Windows (Float Processed)",
#         1: "Building Windows (Non-Float Processed)",
#         2: "Vehicle Windows (Float Processed)",
#         3: "Vehicle Windows (Non-Float Processed)",
#         4: "Containers",
#         5: "Tableware",
#         6: "Headlamps"
#     }

#     st.success(f"### ✅ Predicted Glass Type: **{glass_types[predicted_class]}**")
#     st.write("#### Prediction Probabilities:")
#     st.bar_chart(probs.numpy()[0])

# st.write("---")
# st.write("**📌 Model:** Neural Network (2 Hidden Layers, PyTorch)  |  **Dataset:** UCI Glass Dataset")
# st.write("**🔗 Source Code:** [https://github.com/waheedkhan65/glass_classification_using_NN]")





