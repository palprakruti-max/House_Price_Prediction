
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Housing Price Prediction",
    page_icon="🏠",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = joblib.load("housing_model.pkl")
    features = joblib.load("feature_columns.pkl")
    return model, features

try:
    model, features = load_model()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

st.title("🏠 Housing Price Prediction")
st.write("Enter house details to estimate its price.")

st.subheader("House Details")

house_size = st.number_input(
    "House Size", min_value=0.0, value=1500.0
)
bedrooms = st.number_input(
    "Bedrooms", min_value=0, value=3, step=1
)
bathrooms = st.number_input(
    "Bathrooms", min_value=0.0, value=2.0
)
house_age = st.number_input(
    "House Age (years)", min_value=0, value=10, step=1
)
distance = st.number_input(
    "Distance to City Center (miles)",
    min_value=0.0,
    value=5.0
)
floors = st.number_input(
    "Floors", min_value=1, value=2, step=1
)

if st.button("Predict House Price"):
    input_data = pd.DataFrame(
        [[house_size, bedrooms, bathrooms,
          house_age, distance, floors]],
        columns=[
            "House_Size",
            "bedrooms",
            "bathrooms",
            "House Age",
            "Distance to City Center (miles)",
            "Floors"
        ]
    )

    if list(input_data.columns) != list(features):
        st.error("Input features do not match the trained model.")
        st.stop()

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed!")
    st.metric(
        "Estimated House Price",
        f"${prediction:,.2f}"
    )
