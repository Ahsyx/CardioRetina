# ============================================================
# main.py — CardioRetina v2 Final Backend
# ============================================================
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from PIL import Image
import io, base64, cv2, traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CONFIG
# ============================================================
THRESHOLD  = 0.20
IMG_SIZE   = 240
TTA_STEPS  = 10
W_B1, W_B0 = 0.7, 0.3

# ============================================================
# LOAD MODELS
# ============================================================
def build_model(base_fn, name):
    base    = base_fn(include_top=False, weights=None,
                      input_shape=(IMG_SIZE, IMG_SIZE, 3))
    inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(128, activation='relu')(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name=name)

print("⏳ Loading models...")
dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

try:
    model_b1 = tf.keras.models.load_model("CardioRetina_v2.keras", compile=False)
    model_b0 = tf.keras.models.load_model("CardioRetina_B0.keras", compile=False)
    print("✅ Direct load successful")
except Exception as e:
    print(f"❌ Direct load error: {e}")
    traceback.print_exc()
    print("⚠️  Falling back to weight loading...")
    model_b1 = build_model(tf.keras.applications.EfficientNetB1, 'B1')
    model_b1(dummy)
    model_b1.load_weights("CardioRetina_v2.keras")
    model_b0 = build_model(tf.keras.applications.EfficientNetB0, 'B0')
    model_b0(dummy)
    model_b0.load_weights("CardioRetina_B0.keras")
    print("✅ Weight loading successful")

# ============================================================
# IMAGE QUALITY VALIDATION
# ============================================================
def validate_retinal_scan(img_array):
    img = img_array / 255.0

    if np.mean(img) < 0.05:
        return False, "Invalid Image: Does not match the color profile of a retinal scan. Please upload a valid fundus image."
    if np.mean(img) > 0.97:
        return False, "Invalid Image: Too bright — please upload a properly exposed fundus image."
    if np.std(img) < 0.05:
        return False, "Invalid Image: Lacks the structural complexity of a real eye."

    r_mean = np.mean(img[:, :, 0])
    g_mean = np.mean(img[:, :, 1])
    b_mean = np.mean(img[:, :, 2])
    if b_mean > r_mean and b_mean > g_mean:
        return False, "Invalid Image: Does not match the color profile of a retinal scan. Please upload a valid fundus image."

    h, w = img.shape[:2]
    center  = np.mean(img[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)])
    corners = np.mean([img[0:20,0:20], img[0:20,-20:], img[-20:,0:20], img[-20:,-20:]])
    if corners > center * 1.5:
        return False, "Invalid Image: Illumination pattern does not match a clinical retinal camera."

    return True, "Valid"

# ============================================================
# PREPROCESSING
# ============================================================
def preprocess(image_bytes):
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Preserve aspect ratio with padding
    img_pil.thumbnail((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    padded = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    offset = ((IMG_SIZE - img_pil.width) // 2, (IMG_SIZE - img_pil.height) // 2)
    padded.paste(img_pil, offset)
    arr = np.array(padded, dtype=np.float32)
    return arr, img_pil

# ============================================================
# TTA ENSEMBLE PREDICTION
# ============================================================
def predict_tta(img_array):
    b1_preds, b0_preds = [], []
    for _ in range(TTA_STEPS):
        aug = img_array.copy()
        if np.random.random() > 0.5: aug = np.fliplr(aug)
        if np.random.random() > 0.5: aug = np.flipud(aug)
        delta = np.random.uniform(-0.1, 0.1) * 255
        aug = np.clip(aug + delta, 0, 255)
        factor = np.random.uniform(0.9, 1.1)
        mean = np.mean(aug)
        aug = np.clip((aug - mean) * factor + mean, 0, 255)
        hsv = cv2.cvtColor(aug.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * np.random.uniform(0.9, 1.1), 0, 255)
        aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        batch = np.expand_dims(aug, axis=0)
        b1_preds.append(float(model_b1(batch, training=False).numpy()[0][0]))
        b0_preds.append(float(model_b0(batch, training=False).numpy()[0][0]))
    return (W_B1 * np.mean(b1_preds)) + (W_B0 * np.mean(b0_preds))

# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(img_array):
    img_tensor = tf.cast(np.expand_dims(img_array, 0), tf.float32)
    try:
        efficientnet_layer = model_b1.get_layer('efficientnetb1')
        conv_model = tf.keras.models.Model(
            inputs=efficientnet_layer.input,
            outputs=efficientnet_layer.get_layer('top_conv').output
        )

        # Manually pass through head layers for connected gradient path
        gap   = model_b1.get_layer('global_average_pooling2d_1')
        bn1   = model_b1.get_layer('batch_normalization_2')
        drop1 = model_b1.get_layer('dropout_2')
        den1  = model_b1.get_layer('dense_2')
        bn2   = model_b1.get_layer('batch_normalization_3')
        drop2 = model_b1.get_layer('dropout_3')
        den2  = model_b1.get_layer('dense_3')

        with tf.GradientTape() as tape:
            conv_out = conv_model(img_tensor, training=False)
            tape.watch(conv_out)
            x = gap(conv_out)
            x = bn1(x, training=False)
            x = drop1(x, training=False)
            x = den1(x)
            x = bn2(x, training=False)
            x = drop2(x, training=False)
            preds = den2(x)
            loss  = preds[:, 0]

        grads = tape.gradient(loss, conv_out)

        if grads is None:
            print("❌ Grad-CAM grads None")
            return None, None

        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        original_bgr = cv2.cvtColor(
            img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(
            original_bgr, 0.6, heatmap_colored, 0.4, 0)

        _, buffer = cv2.imencode(".jpg", overlay)
        print("✅ Grad-CAM generated successfully")
        return base64.b64encode(buffer).decode("utf-8"), heatmap_resized

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        traceback.print_exc()
        return None, None
# ============================================================
# CONDITION DETECTION FROM GRAD-CAM ZONES
# ============================================================
def detect_conditions(heatmap, risk_score):
    if risk_score < THRESHOLD:
        return ["No significant vascular abnormalities detected"]
    if heatmap is None:
        return ["Vascular Abnormality Detected"]

    h, w       = heatmap.shape
    conditions = []
    thresh = np.clip(np.mean(heatmap) + np.std(heatmap) * 0.7, 0.03, 0.6)

    if np.mean(heatmap[:h//3, :])                                > thresh: conditions.append("Branch Retinal Vein Occlusion (BRVO)")
    if np.mean(heatmap[2*h//3:, :])                              > thresh: conditions.append("Branch Retinal Artery Occlusion (BRAO)")
    if np.mean(heatmap[h//3:2*h//3, w//3:2*w//3])               > thresh: conditions.append("Central Retinal Involvement (CRVO/CRAO)")
    if np.mean(np.concatenate([heatmap[:, :w//6].flatten(),
                               heatmap[:, 5*w//6:].flatten()]))  > thresh: conditions.append("Hypertensive Retinopathy (HR)")

    if len(conditions) >= 2 and np.mean(heatmap) > thresh * 0.5:
        conditions = ["Diabetic Retinopathy / Hemorrhage (Diffuse)"]

    print(f"🔍 Heatmap stats — mean: {np.mean(heatmap):.3f}, max: {np.max(heatmap):.3f}, zones: top={np.mean(heatmap[:h//3,:]):.3f} bot={np.mean(heatmap[2*h//3:,:]):.3f} center={np.mean(heatmap[h//3:2*h//3,w//3:2*w//3]):.3f} thresh={thresh:.3f}")

    if not conditions and np.mean(heatmap) > 0.15:
        conditions = ["Diffuse Vascular Changes Detected"]

    if not conditions:
        conditions = ["Vascular Abnormality Detected"]

    return conditions
# ============================================================
# RECOMMENDATION
# ============================================================
def get_recommendation(score):
    if score >= 0.80:
        return "Patient shows signs of high cardiovascular risk. Immediate blood pressure screening and cardiology consultation is strongly recommended."
    elif score >= 0.50:
        return "Moderate cardiovascular risk indicators detected. Schedule a follow-up with your physician within 2-4 weeks for further evaluation."
    elif score >= 0.20:
        return "Mild cardiovascular risk indicators present. Monitor blood pressure regularly and consult your doctor at your next scheduled visit."
    else:
        return "Retinal scan appears within normal parameters. Routine annual screening recommended. Maintain healthy blood pressure and cholesterol levels."

# ============================================================
# IMAGE TO BASE64
# ============================================================
def image_to_base64(img_pil):
    img_resized = img_pil.resize((400, 400))
    buffer      = io.BytesIO()
    img_resized.save(buffer, format="JPEG", quality=75)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ============================================================
# ROUTES
# ============================================================
@app.get("/")
def root():
    return {
        "status"   : "CardioRetina v2 API is running",
        "model"    : "EfficientNetB1 + B0 Ensemble + TTA",
        "threshold": THRESHOLD,
        "version"  : "2.0"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img_array, img_pil = preprocess(image_bytes)

        # Validate
        is_valid, err_msg = validate_retinal_scan(img_array)
        if not is_valid:
            return JSONResponse({"error": err_msg}, status_code=400)

        # TTA Ensemble
        score        = predict_tta(img_array)
        label        = "High Risk" if score >= THRESHOLD else "Low Risk"
        risk_percent = round(float(score) * 100, 2)

        # Grad-CAM
        gradcam_b64, heatmap = None, None
        try:
            gradcam_b64, heatmap = generate_gradcam(img_array)
        except Exception as e:
            print(f"Grad-CAM failed: {e}")

        # Conditions
        conditions = detect_conditions(heatmap, score)

        # Recommendation
        recommendation = get_recommendation(score)

        # Original image base64
        image_b64 = image_to_base64(img_pil)

        # ── Response matches your frontend exactly ──
        return JSONResponse({
            "risk_score"    : risk_percent,        # data.risk_score ✅
            "label"         : label,               # data.label ✅
            "confidence"    : round(score, 4),     # data.confidence ✅
            "conditions"    : conditions,          # data.conditions ✅
            "recommendation": recommendation,      # data.recommendation ✅
            "gradcam"       : gradcam_b64,         # data.gradcam ✅
            "image_b64"     : image_b64,           # data.image_b64 ✅
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
    


