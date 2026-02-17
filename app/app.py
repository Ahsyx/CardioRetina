import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# --- CONFIGURATION & PATHS ---
IMAGE_RES = (224, 224)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'CardioRetina_Model.keras')
META_PATH = os.path.join(BASE_DIR, '..', 'models', 'model_meta.json')

st.set_page_config(page_title="CardioRetina", page_icon="👁️", layout="centered")

def inject_styles():
    st.markdown("""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Dancing+Script:wght@400..700&family=Poppins:wght@400;700&display=swap" rel="stylesheet">
        <style>
            [data-testid="stAppViewContainer"] { background-color: #FF8552 !important; }
            [data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
            [data-testid="stSidebar"] { display: none; }

            .block-container {
                background-color: #FFFCFF !important;
                padding: 30px 40px !important; 
                border-radius: 30px !important;
                box-shadow: 15px 15px 0px rgba(0,0,0,1) !important;
                margin-top: 60px !important;
                max-width: 800px !important;
                min-height: 520px; 
            }

            .poppins-font { font-family: 'Poppins', sans-serif !important; font-weight: 700; font-size: 48px; color: #31333F !important; }
            .dancing-font { font-family: 'Dancing Script', cursive !important; font-size: 56px; color: #C2AFF0 !important; margin-left: 5px; }

            [data-testid="stFileUploader"] section {
                display: flex !important; justify-content: center !important; align-items: center !important;
                height: 180px !important; background-color: rgba(49, 51, 63, 0.05) !important;
                border: 2px dashed #C2AFF0 !important; border-radius: 15px !important;
            }
            [data-testid="stFileUploader"] section > div, [data-testid="stFileUploaderDropzone"] + div { display: none !important; }

            .result-container { display: flex; align-items: baseline; justify-content: center; gap: 8px; margin: 5px 0px !important; }
            .risk-label-poppins { font-family: 'Poppins', sans-serif !important; font-size: 22px; text-transform: lowercase; }
            .risk-percent { font-family: 'Poppins', sans-serif !important; font-size: 16px; color: #666; }
            .risk-high-text { color: #d9534f !important; }
            .risk-low-text { color: #5cb85c !important; }

            .stMarkdown, .stTitle, .stText, .stImage, [data-testid="stVerticalBlock"], .stButton {
                text-align: center !important; display: flex; flex-direction: column; align-items: center; justify-content: center;
            }
            h1, h2, h3, p, span, label, .stMarkdown { color: #31333F !important; }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_assets():
    try:
        with open(META_PATH, 'r') as f:
            meta = json.load(f)
        
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception:
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras import layers, models
            base = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
            model = models.Sequential([base, layers.GlobalAveragePooling2D(), layers.Dropout(0.3), layers.Dense(1, activation='sigmoid')])
            model.load_weights(MODEL_PATH)
        return model, meta
    except Exception as e:
        st.error(f"Critical error loading assets: {e}")
        return None, None

def handle_restart():
    st.session_state.uploader_key += 1

def main():
    inject_styles()
    model, _ = load_model_assets()
    
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    st.markdown('<h1><span class="poppins-font">Cardio</span><span class="dancing-font">Retina</span></h1>', unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; color: #31333F; max-width: 600px; margin: 0 auto; line-height: 1.4; font-size: 14px; padding-bottom: 15px;">
            CardioRetina utilizes deep learning to identify subtle cardiovascular markers within retinal vasculature. 
            By detecting hypertensive retinopathy and vessel occlusions, it provides a vital early screening layer. 
            This prototype prioritizes clinical safety, achieving a high-sensitivity recall rate of 88.5%.
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='margin: 10px 0px; border-color: #eee;'>", unsafe_allow_html=True)

    ui_placeholder = st.empty()
    with ui_placeholder.container():
        file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed", key=f"uploader_{st.session_state.uploader_key}")
        if not file:
            st.caption("JPG, PNG, or JPEG • Max 200MB")

    if file and model:
        ui_placeholder.empty()
        img = Image.open(file).convert("RGB")
        st.image(img, width=220)
        
        processed_img = np.expand_dims(np.array(img.resize(IMAGE_RES)) / 255.0, axis=0)
        prediction = float(model.predict(processed_img, verbose=0).flatten()[0])
        
        is_high = prediction > 0.5
        label = "high risk" if is_high else "low risk"
        status_class = "risk-high-text" if is_high else "risk-low-text"
        
        st.markdown(f"""
            <div class="result-container">
                <span class="risk-label-poppins {status_class}">{label}</span>
                <span class="risk-percent">({prediction*100:.1f}%)</span>
            </div>
        """, unsafe_allow_html=True)
        
        if is_high:
            st.warning("⚠️ Pathological markers detected. Consultation recommended.")
        else:
            st.success("✅ No significant cardiovascular markers found.")

        st.button("Analyze Another Scan", on_click=handle_restart)

    st.markdown("<hr style='margin: 10px 0px; border-color: #eee;'>", unsafe_allow_html=True)
    st.caption("AI Prototype: Not a substitute for professional medical diagnosis.")

if __name__ == "__main__":
    main()