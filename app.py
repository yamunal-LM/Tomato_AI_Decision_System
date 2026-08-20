import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Tomato AI Decision System",
    page_icon="🍅",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🍅 Tomato AI Decision System")
st.write(
    "Integrated AI system for Tomato Disease Detection, "
    "Nutrient Deficiency, Soil Health and Fertilizer Recommendation."
)

st.divider()

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():

    # Disease model
    interpreter = tf.lite.Interpreter(
        model_path="tomato_leaf_model.tflite"
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Nutrient model
    nutrient_model = joblib.load("nutrient_model.pkl")
    nutrient_scaler = joblib.load("nutrient_scaler.pkl")
    nutrient_features = joblib.load("nutrient_feature_columns.pkl")

    # Soil model
    soil_model = joblib.load("soil_model.pkl")
    soil_scaler = joblib.load("soil_scaler.pkl")
    soil_features = joblib.load("soil_feature_columns.pkl")

    # Fertilizer model
    fertilizer_model = joblib.load("fertilizer_model.pkl")
    fertilizer_scaler = joblib.load("fertilizer_scaler.pkl")
    fertilizer_features = joblib.load(
        "fertilizer_feature_columns.pkl"
    )

    return (
        interpreter,
        input_details,
        output_details,
        nutrient_model,
        nutrient_scaler,
        nutrient_features,
        soil_model,
        soil_scaler,
        soil_features,
        fertilizer_model,
        fertilizer_scaler,
        fertilizer_features
    )


try:
    (
        interpreter,
        input_details,
        output_details,
        nutrient_model,
        nutrient_scaler,
        nutrient_features,
        soil_model,
        soil_scaler,
        soil_features,
        fertilizer_model,
        fertilizer_scaler,
        fertilizer_features
    ) = load_models()

except Exception as e:
    st.error("⚠️ Model loading error.")
    st.error(str(e))
    st.stop()


# --------------------------------------------------
# DISEASE CLASS NAMES
# --------------------------------------------------
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


# ==================================================
# INPUT SECTION
# ==================================================

st.header("📋 Tomato Analysis Input")

# --------------------------------------------------
# LEAF IMAGE
# --------------------------------------------------
st.subheader("🦠 1. Tomato Leaf Disease Detection")

uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# NUTRIENT INPUT
# --------------------------------------------------
st.subheader("🧪 2. Nutrient Deficiency Details")

col1, col2 = st.columns(2)

with col1:

    leaf_color = st.selectbox(
        "Leaf Color",
        [
            "Green with Purple Tint",
            "Light Green",
            "Dark Green",
            "Pale Green",
            "Yellow Edge",
            "Interveinal Yellow"
        ]
    )

    yellowing = st.selectbox(
        "Yellowing",
        ["Yes", "No"]
    )

    brown_spots = st.selectbox(
        "Brown Spots",
        ["Yes", "No"]
    )

with col2:

    leaf_curl = st.selectbox(
        "Leaf Curl",
        [
            "Slight",
            "Moderate",
            "NaN"
        ]
    )

    vein_color = st.selectbox(
        "Vein Color",
        [
            "Green",
            "Yellow",
            "Purple"
        ]
    )


# --------------------------------------------------
# SOIL INPUT
# --------------------------------------------------
st.subheader("🌱 3. Soil Health Details")

col1, col2, col3 = st.columns(3)

with col1:

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

with col2:

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

with col3:

    organic_matter = st.number_input(
        "Organic Matter (%)",
        value=4.2
    )

    application_rate = st.number_input(
        "Application Rate (kg/acre)",
        value=50.0
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


st.divider()


# ==================================================
# ONE BUTTON
# ==================================================

analyze = st.button(
    "🔍 ANALYZE TOMATO",
    use_container_width=True
)


# ==================================================
# ANALYSIS
# ==================================================

if analyze:

    # --------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------

    if uploaded_file is None:

        st.warning(
            "⚠️ Please upload a tomato leaf image."
        )

    else:

        # ==================================================
        # 1. DISEASE PREDICTION
        # ==================================================

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.subheader(
            "🦠 Tomato Disease Detection Result"
        )

        st.image(
            image,
            caption="Uploaded Tomato Leaf",
            use_container_width=True
        )

        img = image.resize((224, 224))

        img = np.array(
            img,
            dtype=np.float32
        )

        img = img / 255.0

        img = np.expand_dims(
            img,
            axis=0
        )

        interpreter.set_tensor(
            input_details[0]["index"],
            img
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details[0]["index"]
        )

        predicted_index = int(
            np.argmax(output)
        )

        confidence = float(
            np.max(output) * 100
        )

        disease_prediction = class_names[
            predicted_index
        ]

        col1, col2 = st.columns(2)

        with col1:
            st.success(
                f"🦠 Disease: {disease_prediction}"
            )

        with col2:
            st.info(
                f"🎯 Confidence: {confidence:.2f}%"
            )


        # ==================================================
        # 2. NUTRIENT DEFICIENCY
        # ==================================================

        st.subheader(
            "🧪 Nutrient Deficiency Result"
        )

        nutrient_data = pd.DataFrame({

            "Sample_ID": ["TOM0001"],

            "Leaf_Color": [leaf_color],

            "Yellowing": [yellowing],

            "Brown_Spots": [brown_spots],

            "Leaf_Curl": [
                np.nan if leaf_curl == "NaN"
                else leaf_curl
            ],

            "Vein_Color": [vein_color]
        })

        nutrient_data = pd.get_dummies(
            nutrient_data
        )

        nutrient_data = nutrient_data.reindex(
            columns=nutrient_features,
            fill_value=0
        )

        nutrient_data = nutrient_scaler.transform(
            nutrient_data
        )

        nutrient_prediction = nutrient_model.predict(
            nutrient_data
        )

        nutrient_result = nutrient_prediction[0]

        st.success(
            f"🧪 Predicted Nutrient Deficiency: "
            f"{nutrient_result}"
        )


        # ==================================================
        # 3. SOIL HEALTH
        # ==================================================

        st.subheader(
            "🌱 Soil Health Result"
        )

        soil_data = pd.DataFrame({

            "Sample_ID": [101],

            "Soil_pH": [soil_ph],

            "Moisture_%": [moisture],

            "Nitrogen_mgkg": [nitrogen],

            "Phosphorus_mgkg": [phosphorus],

            "Potassium_mgkg": [potassium],

            "Temperature_C": [temperature],

            "Organic_Matter_%": [organic_matter]
        })

        soil_data = pd.get_dummies(
            soil_data
        )

        soil_data = soil_data.reindex(
            columns=soil_features,
            fill_value=0
        )

        soil_data = soil_scaler.transform(
            soil_data
        )

        soil_prediction = soil_model.predict(
            soil_data
        )

        soil_result = soil_prediction[0]

        st.success(
            f"🌱 Predicted Soil Health: "
            f"{soil_result}"
        )


        # ==================================================
        # 4. FERTILIZER RECOMMENDATION
        # ==================================================

        st.subheader(
            "🧴 Fertilizer Recommendation"
        )

        fertilizer_data = pd.DataFrame({

            "Sample_ID": [1],

            "Soil_pH": [soil_ph],

            "Nitrogen_mgkg": [nitrogen],

            "Phosphorus_mgkg": [phosphorus],

            "Potassium_mgkg": [potassium],

            "Moisture_%": [moisture],

            "Growth_Stage": [growth_stage],

            "Application_Rate_kg_per_acre": [
                application_rate
            ]
        })

        fertilizer_data = pd.get_dummies(
            fertilizer_data
        )

        fertilizer_data = fertilizer_data.reindex(
            columns=fertilizer_features,
            fill_value=0
        )

        fertilizer_data = fertilizer_scaler.transform(
            fertilizer_data
        )

        fertilizer_prediction = fertilizer_model.predict(
            fertilizer_data
        )

        fertilizer_result = fertilizer_prediction[0]

        st.success(
            f"🧴 Recommended Fertilizer: "
            f"{fertilizer_result}"
        )


        # ==================================================
        # FINAL AI DECISION
        # ==================================================

        st.divider()

        st.header(
            "🤖 Final AI Decision"
        )

        st.write(
            "The four AI modules have been analyzed "
            "together."
        )

        st.info(
            f"""
🦠 **Disease:** {disease_prediction}

🧪 **Nutrient Deficiency:** {nutrient_result}

🌱 **Soil Health:** {soil_result}

🧴 **Recommended Fertilizer:** {fertilizer_result}
"""
        )

        st.success(
            "🌱 AI analysis completed successfully!"
        )
