
import streamlit as st
import tensorflow as tf
import numpy as np
import json

from PIL import Image
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "plant_disease_cnn.keras"
CLASS_PATH = BASE_DIR / "class_names.json"


# ==========================================
# 2. CUSTOM CSS DESIGN
# ==========================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #f1f7f1;
    color: #26372b;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Header */
.main-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    color: #176b3a !important;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #52665a !important;
    font-size: 17px;
    margin-bottom: 25px;
}

/* Force readable text */
h1, h2, h3, h4, h5, h6,
p, label, li,
[data-testid="stMarkdownContainer"] {
    color: #26372b;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #e1eee2;
}

/* Cards */
.result-card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 16px;
    border-left: 6px solid #238b50;
    box-shadow: 0 3px 12px rgba(0,0,0,0.07);
    margin-top: 15px;
    margin-bottom: 15px;
}

.result-card h2,
.result-card h3,
.result-card p {
    color: #176b3a !important;
}

/* Information cards */
.info-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #dce9dc;
    margin-bottom: 15px;
}

/* Section headings */
.section-heading {
    color: #176b3a !important;
    font-size: 23px;
    font-weight: 700;
}

/* Buttons */
.stButton button {
    background-color: #238b50;
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 600;
    width: 100%;
}

.stButton button:hover {
    background-color: #176b3a;
    color: white !important;
    border: none;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    border: 1px dashed #238b50;
}

/* Uploaded file text */
[data-testid="stFileUploader"] * {
    color: #26372b !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background-color: #238b50;
}

/* Footer */
.footer {
    text-align: center;
    color: #647568 !important;
    font-size: 14px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🌱 PlantCare AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Smart Plant Disease Detection Using Deep Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 4. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🌿 PlantCare AI")

    st.write(
        "An intelligent system that identifies plant leaf "
        "diseases using Convolutional Neural Networks."
    )

    st.divider()

    st.markdown("### 🌱 Supported Plants")

    st.write("🍅 Tomato")
    st.write("🥔 Potato")
    st.write("🫑 Bell Pepper")

    st.divider()

    st.markdown("### ⚙️ Technologies")

    st.write("• Python")
    st.write("• TensorFlow / Keras")
    st.write("• CNN")
    st.write("• Streamlit")
    st.write("• NumPy")
    st.write("• Pillow")

    st.divider()

    st.caption("Minor Project | 2026")


# ==========================================
# 5. DISEASE INFORMATION
# ==========================================

disease_info = {

    "Pepper,_bell___Bacterial_spot": {
        "name": "Bell Pepper - Bacterial Spot",
        "symptoms": "Small dark, water-soaked spots on leaves and fruits.",
        "prevention": "Use disease-free seeds, avoid overhead watering, "
                      "and remove infected plant material."
    },

    "Pepper,_bell___healthy": {
        "name": "Bell Pepper - Healthy",
        "symptoms": "Green leaves without visible disease symptoms.",
        "prevention": "Maintain proper watering, sunlight, and nutrition."
    },

    "Potato___Early_blight": {
        "name": "Potato - Early Blight",
        "symptoms": "Brown circular spots with concentric rings on older leaves.",
        "prevention": "Remove infected leaves, rotate crops, and maintain spacing."
    },

    "Potato___healthy": {
        "name": "Potato - Healthy",
        "symptoms": "Leaves appear green and free from visible disease.",
        "prevention": "Maintain soil health and monitor plants regularly."
    },

    "Potato___Late_blight": {
        "name": "Potato - Late Blight",
        "symptoms": "Dark, water-soaked lesions that can spread rapidly.",
        "prevention": "Use healthy seed potatoes and remove infected plant material."
    },

    "Tomato___Bacterial_spot": {
        "name": "Tomato - Bacterial Spot",
        "symptoms": "Small dark spots on leaves, stems, and fruits.",
        "prevention": "Use clean seeds, avoid overhead irrigation, "
                      "and remove infected plant material."
    },

    "Tomato___Early_blight": {
        "name": "Tomato - Early Blight",
        "symptoms": "Brown spots with target-like rings, usually on older leaves.",
        "prevention": "Remove affected leaves, rotate crops, and maintain spacing."
    },

    "Tomato___healthy": {
        "name": "Tomato - Healthy",
        "symptoms": "Leaves are green without visible disease symptoms.",
        "prevention": "Continue regular watering and provide adequate sunlight."
    },

    "Tomato___Late_blight": {
        "name": "Tomato - Late Blight",
        "symptoms": "Large dark patches on leaves, sometimes with white growth underneath.",
        "prevention": "Improve air circulation and remove infected plant material."
    },

    "Tomato___Leaf_Mold": {
        "name": "Tomato - Leaf Mold",
        "symptoms": "Yellow patches on upper leaves and olive-green growth underneath.",
        "prevention": "Improve ventilation and reduce excessive humidity."
    },

    "Tomato___Septoria_leaf_spot": {
        "name": "Tomato - Septoria Leaf Spot",
        "symptoms": "Small circular spots with dark borders and pale centers.",
        "prevention": "Remove infected leaves, mulch soil, and rotate crops."
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Tomato - Spider Mites",
        "symptoms": "Tiny yellow speckles, leaf discoloration, and fine webbing.",
        "prevention": "Inspect leaf undersides and manage mite populations."
    },

    "Tomato___Target_Spot": {
        "name": "Tomato - Target Spot",
        "symptoms": "Circular brown lesions with concentric rings.",
        "prevention": "Improve airflow and remove infected leaves."
    },

    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato - Mosaic Virus",
        "symptoms": "Mottled light and dark green leaves and leaf distortion.",
        "prevention": "Remove infected plants and sanitize gardening tools."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato - Yellow Leaf Curl Virus",
        "symptoms": "Yellowing, upward curling leaves, and stunted plant growth.",
        "prevention": "Control whiteflies and use resistant plant varieties."
    }
}


# ==========================================
# 6. LOAD TRAINED CNN MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        str(MODEL_PATH)
    )

    with open(CLASS_PATH, "r") as file:
        class_names = json.load(file)

    return model, class_names


try:

    model, class_names = load_model()

    st.success("✅ AI Model loaded successfully!")

except Exception as e:

    st.error(f"Model loading error: {e}")
    st.stop()


# ==========================================
# 7. IMAGE UPLOAD SECTION
# ==========================================

st.markdown(
    '<div class="section-heading">📷 Upload Plant Leaf Image</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a clear image of a tomato, potato, or bell pepper leaf."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# 8. IMAGE DISPLAY AND PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:

        st.markdown("### 🖼️ Uploaded Image")

        st.image(
            image,
            caption="Uploaded Plant Leaf",
            use_container_width=True
        )

    with col2:

        st.markdown("### 🔍 Disease Analysis")

        st.write(
            "Click below to analyze the uploaded leaf image."
        )

        predict_button = st.button(
            "🌱 Detect Plant Disease",
            type="primary"
        )

    if predict_button:

        with st.spinner("Analyzing plant leaf using CNN model..."):

            # Image preprocessing
            resized_image = image.resize((128, 128))

            image_array = np.array(
                resized_image,
                dtype=np.float32
            ) / 255.0

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # CNN prediction
            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            top_indices = np.argsort(predictions)[::-1][:3]

            predicted_index = top_indices[0]

            predicted_class = class_names[predicted_index]

            confidence = float(
                predictions[predicted_index]
            ) * 100

            info = disease_info.get(
                predicted_class,
                {
                    "name": predicted_class.replace("_", " "),
                    "symptoms": "No additional information available.",
                    "prevention": "Consult an agricultural expert."
                }
            )

        # ==================================
        # 9. MAIN RESULT
        # ==================================

        st.divider()

        st.markdown("## 📊 Disease Detection Result")

        result_col1, result_col2 = st.columns([2, 1])

        with result_col1:

            st.markdown(
                f"""
                <div class="result-card">
                    <h3>🌿 Predicted Condition</h3>
                    <h2>{info["name"]}</h2>
                    <p>Model confidence: <b>{confidence:.2f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col2:

            st.markdown(
                f"""
                <div class="result-card">
                    <h3>🎯 Confidence</h3>
                    <h2>{confidence:.2f}%</h2>
                    <p>CNN Prediction Score</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(
            float(predictions[predicted_index])
        )

        # ==================================
        # 10. SYMPTOMS AND PREVENTION
        # ==================================

        st.markdown("## 🩺 Disease Information")

        info_col1, info_col2 = st.columns(2)

        with info_col1:

            st.markdown(
                f"""
                <div class="info-card">
                    <h3>🩺 Symptoms</h3>
                    <p>{info["symptoms"]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with info_col2:

            st.markdown(
                f"""
                <div class="info-card">
                    <h3>🛡️ Prevention</h3>
                    <p>{info["prevention"]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ==================================
        # 11. TOP THREE PREDICTIONS
        # ==================================

        st.markdown("## 🏆 Top 3 Predictions")

        for rank, index in enumerate(top_indices, start=1):

            label = class_names[index].replace("_", " ")

            score = float(predictions[index])

            st.write(
                f"**{rank}. {label} — {score * 100:.2f}%**"
            )

            st.progress(score)

        st.caption(
            "Note: Confidence represents the model's prediction score, "
            "not a guarantee of diagnostic accuracy. "
            "Consult an agricultural expert before treatment."
        )

# ==========================================
# 12. ABOUT PROJECT
# ==========================================

st.divider()

st.markdown("## 🌱 About PlantCare AI")

st.write("""
PlantCare AI is a deep learning-based plant disease detection
application developed using Convolutional Neural Networks (CNN).

The system analyzes uploaded leaf images and predicts possible
diseases in tomato, potato, and bell pepper plants.

The model is trained using the PlantVillage dataset and supports
15 different healthy and diseased leaf categories.
""")

# ==========================================
# 13. PROJECT DETAILS
# ==========================================

st.markdown("## ⚙️ Project Details")

detail1, detail2, detail3 = st.columns(3)

with detail1:

    st.markdown(
        """
        <div class="info-card">
            <h3>🧠 Algorithm</h3>
            <p>Convolutional Neural Network (CNN)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with detail2:

    st.markdown(
        """
        <div class="info-card">
            <h3>📂 Dataset</h3>
            <p>PlantVillage Dataset</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with detail3:

    st.markdown(
        """
        <div class="info-card">
            <h3>🌿 Categories</h3>
            <p>15 Leaf Categories</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <h3 style="color:#176b3a;">🌱 PlantCare AI</h3>
        <p>Minor Project | B.Tech CSE (AI & ML) | 2026</p>
        <p>Developed by Anshika Sethi, Sakshi Singh & Payal</p>
        <p>Powered by Deep Learning and Computer Vision</p>
    </div>
    """,
    unsafe_allow_html=True
)
