import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Tomato AI Decision System",
    page_icon="🍅",
    layout="wide"
)

st.title("🍅 Tomato AI Decision System")
st.write(
    "An integrated AI system for Tomato Leaf Disease, "
    "Nutrient Deficiency, Soil Health and Fertilizer Recommendation."
)

st.divider()


# =========================================================
# SIDEBAR - MODULE SELECTION
# =========================================================

st.sidebar.title("🌱 AI Modules")

module = st.sidebar.radio(
    "Select Module",
    [
        "🍃 Leaf Disease Detection",
        "🧪 Nutrient Deficiency",
        "🌱 Soil Health Prediction",
        "🧴 Fertilizer Recommendation"
    ]
)


# =========================================================
# 1. LEAF DISEASE DETECTION
# =========================================================

if module == "🍃 Leaf Disease Detection":

    st.header("🍃 Tomato Leaf Disease Detection")

    st.write("Upload a tomato leaf image to predict the disease.")

    uploaded_file = st.file_uploader(
        "Choose a tomato leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Tomato Leaf",
            use_container_width=True
        )

        if st.button("🔍 Predict Disease"):

            interpreter = tf.lite.Interpreter(
                model_path="tomato_leaf_model.tflite"
            )

            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            img = image.resize((224, 224))
            img = np.array(img, dtype=np.float32)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            interpreter.set_tensor(
                input_details[0]["index"],
                img
            )

            interpreter.invoke()

            output = interpreter.get_tensor(
                output_details[0]["index"]
            )

            predicted_index = np.argmax(output)
            confidence = float(np.max(output) * 100)

            class_names = [
                "Tomato_Bacterial_spot",
                "Tomato_Early_blight",
                "Tomato_Late_blight",
                "Tomato_Leaf_Mold",
                "Tomato_Septoria_leaf_spot",
                "Tomato_Spider_mites_Two_spotted_spider_mite",
                "Tomato_Target_Spot",
                "Tomato_Tomato_mosaic_virus",
                "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
                "Tomato_healthy"
            ]

            st.success(
                f"🍃 Prediction: {class_names[predicted_index]}"
            )

            st.info(
                f"Confidence: {confidence:.2f}%"
            )


# =========================================================
# 2. NUTRIENT DEFICIENCY
# =========================================================

elif module == "🧪 Nutrient Deficiency":

    st.header("🧪 Tomato Nutrient Deficiency Prediction")

    st.write("Enter the tomato leaf details below.")

    sample_id = st.number_input(
        "Sample ID",
        value=201
    )

    leaf_color = st.selectbox(
        "Leaf Color",
        ["Green", "Yellow", "Light Green", "Dark Green"]
    )

    yellowing = st.selectbox(
        "Yellowing",
        ["Low", "Medium", "High"]
    )

    brown_spots = st.selectbox(
        "Brown Spots",
        ["Yes", "No"]
    )

    leaf_curl = st.selectbox(
        "Leaf Curl",
        ["Yes", "No"]
    )

    vein_color = st.selectbox(
        "Vein Color",
        ["Green", "Yellow", "Light Green"]
    )

    if st.button("🔍 Predict Nutrient Deficiency"):

        model = joblib.load("nutrient_model.pkl")
        scaler = joblib.load("nutrient_scaler.pkl")
        feature_columns = joblib.load(
            "nutrient_feature_columns.pkl"
        )

        new_data = pd.DataFrame({
            "Sample_ID": [sample_id],
            "Leaf_Color": [leaf_color],
            "Yellowing": [yellowing],
            "Brown_Spots": [brown_spots],
            "Leaf_Curl": [leaf_curl],
            "Vein_Color": [vein_color]
        })

        new_data = pd.get_dummies(new_data)

        new_data = new_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        new_data = scaler.transform(new_data)

        prediction = model.predict(new_data)

        st.success(
            f"🧪 Predicted Nutrient Deficiency: {prediction[0]}"
        )


# =========================================================
# 3. SOIL HEALTH PREDICTION
# =========================================================

elif module == "🌱 Soil Health Prediction":

    st.header("🌱 Tomato Soil Health Prediction")

    st.write("Enter the soil details below.")

    sample_id = st.number_input(
        "Sample ID",
        value=101
    )

    soil_ph = st.number_input(
        "Soil pH",
        value=6.8
    )

    moisture = st.number_input(
        "Moisture (%)",
        value=65.0
    )

    nitrogen = st.number_input(
        "Nitrogen (mg/kg)",
        value=45.0
    )

    phosphorus = st.number_input(
        "Phosphorus (mg/kg)",
        value=30.0
    )

    potassium = st.number_input(
        "Potassium (mg/kg)",
        value=180.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        value=27.0
    )

    organic_matter = st.number_input(
        "Organic Matter (%)",
        value=4.2
    )

    if st.button("🔍 Predict Soil Health"):

        model = joblib.load("soil_model.pkl")
        scaler = joblib.load("soil_scaler.pkl")
        feature_columns = joblib.load(
            "soil_feature_columns.pkl"
        )

        new_data = pd.DataFrame({
            "Sample_ID": [sample_id],
            "Soil_pH": [soil_ph],
            "Moisture_%": [moisture],
            "Nitrogen_mgkg": [nitrogen],
            "Phosphorus_mgkg": [phosphorus],
            "Potassium_mgkg": [potassium],
            "Temperature_C": [temperature],
            "Organic_Matter_%": [organic_matter]
        })

        new_data = pd.get_dummies(new_data)

        new_data = new_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        new_data = scaler.transform(new_data)

        prediction = model.predict(new_data)

        st.success(
            f"🌱 Predicted Soil Health: {prediction[0]}"
        )


# =========================================================
# 4. FERTILIZER RECOMMENDATION
# =========================================================

elif module == "🧴 Fertilizer Recommendation":

    st.header("🧴 Tomato Fertilizer Recommendation")

    st.write("Enter the soil and crop details below.")

    soil_ph = st.number_input(
        "Soil pH",
        value=6.5
    )

    nitrogen = st.number_input(
        "Nitrogen (mg/kg)",
        value=45
    )

    phosphorus = st.number_input(
        "Phosphorus (mg/kg)",
        value=30
    )

    potassium = st.number_input(
        "Potassium (mg/kg)",
        value=180
    )

    moisture = st.number_input(
        "Moisture (%)",
        value=65
    )

    application_rate = st.number_input(
        "Application Rate (kg/acre)",
        value=50
    )

    growth_stage = st.selectbox(
        "Growth Stage",
        [
            "Seedling",
            "Vegetative",
            "Flowering",
            "Fruiting"
        ]
    )

    if st.button("🔍 Predict Fertilizer"):

        model = joblib.load("fertilizer_model.pkl")
        scaler = joblib.load("fertilizer_scaler.pkl")
        feature_columns = joblib.load(
            "fertilizer_feature_columns.pkl"
        )

        data = pd.DataFrame({
            "Sample_ID": [1],
            "Soil_pH": [soil_ph],
            "Nitrogen_mgkg": [nitrogen],
            "Phosphorus_mgkg": [phosphorus],
            "Potassium_mgkg": [potassium],
            "Moisture_%": [moisture],
            "Growth_Stage": [growth_stage],
            "Application_Rate_kg_per_acre": [application_rate]
        })

        data = pd.get_dummies(data)

        data = data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        data = scaler.transform(data)

        prediction = model.predict(data)

        st.success(
            f"🧴 Recommended Fertilizer: {prediction[0]}"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🍅 Tomato AI Decision System | "
    "AI-based Agricultural Decision Support"
)
