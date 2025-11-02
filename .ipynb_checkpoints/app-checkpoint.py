import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model("ids_model.h5")

st.title("🛡️ AI-Based Network Intrusion Detection System")
st.write("Upload your network data or generate sample inputs to detect if the traffic is normal or an attack.")

# --- Option 1: Upload a CSV file ---
uploaded_file = st.file_uploader("📂 Upload a CSV file with 122 features", type=["csv"])

# --- Option 2: Generate random example ---
use_example = st.checkbox("Use random example input (auto-fill 122 features)")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if data.shape[1] != 122:
        st.error(f"❌ Expected 122 features, but got {data.shape[1]}. Please upload the correct file.")
    else:
        st.success("✅ File uploaded successfully!")
        prediction = model.predict(data)
        result = ["🚨 Attack" if p > 0.5 else "✅ Normal" for p in prediction]
        st.write(pd.DataFrame({"Prediction": result}))
elif use_example:
    # Generate random feature values for testing
    input_data = np.random.rand(1, 122)
    st.write("Generated input sample (first 10 features):", input_data[0, :10])
    prediction = model.predict(input_data)[0][0]
    if prediction > 0.5:
        st.error("🚨 Attack detected!")
    else:
        st.success("✅ Normal traffic detected.")
else:
    st.info("👆 Upload a CSV file or check the box to use random example data.")
