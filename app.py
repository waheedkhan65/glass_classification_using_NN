import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import joblib
from src.nn_model_architecture import Model

# ----------------------
# Paths
# ----------------------
MODEL_PATH = "models/my_nn_model.pt"
SCALER_PATH = "models/scaler.pkl"

# ----------------------
# Load model & scaler
# ----------------------
scaler = joblib.load(SCALER_PATH)
model = Model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Glass Classification", page_icon="🔍", layout="centered")
st.title("Glass Type Classification using Neural Network 🔍")

st.write("""
This app predicts the **type of glass** based on its chemical composition using a trained Neural Network model.
Provide the feature values below:
""")

feature_names = [
    "RI: Refractive Index", "Na: Sodium", "Mg: Magnesium", "Al: Aluminum",
    "Si: Silicon", "K: Potassium", "Ca: Calcium", "Ba: Barium", "Fe: Iron"
]

# Collect inputs
input_data = []
cols = st.columns(3)
for i, feature in enumerate(feature_names):
    value = cols[i % 3].number_input(f"{feature}", min_value=0.0, max_value=20.0, value=1.0, step=0.01)
    input_data.append(value)

# Predict
if st.button("🔍 Predict Glass Type"):
    # Preprocess
    X_input = np.array(input_data).reshape(1, -1)
    X_scaled = scaler.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Prediction
    with torch.no_grad():
        output = model(X_tensor)
        probs = F.softmax(output, dim=1).numpy()[0]
        predicted_class = int(np.argmax(probs))

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

    # Show probabilities as a DataFrame for better visualization
    import pandas as pd
    prob_df = pd.DataFrame({"Glass Type": list(glass_types.values()), "Probability": probs})
    st.bar_chart(prob_df.set_index("Glass Type"))

st.write("---")
st.write("**📌 Model:** Neural Network (5 Hidden Layers, PyTorch)  |  **Dataset:** UCI Glass Dataset")
st.write("**🔗 Source Code:** [https://github.com/waheedkhan65/glass_classification_using_NN]")
