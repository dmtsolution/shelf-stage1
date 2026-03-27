import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort
import os
import time
import io
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Shelf Recognition — Stage 1",
    page_icon="🛒",
    layout="wide"
)

CLASS_COLORS = {
    'boisson_energetique': (231, 76, 60),
    'dessert': (243, 156, 18),
    'eau': (52, 152, 219),
    'fromage': (241, 196, 15),
    'jus': (230, 126, 34),
    'lait': (189, 195, 199),
    'soda': (233, 30, 99),
    'yaourt': (155, 89, 182),
}

# Noms des classes (à adapter selon votre modèle)
CLASS_NAMES = list(CLASS_COLORS.keys())

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best.onnx")
    if not os.path.exists(model_path):
        st.error(f"❌ Modèle ONNX introuvable: {model_path}")
        st.info("💡 Convertissez votre modèle .pt en .onnx avec: model.export(format='onnx')")
        return None
    try:
        session = ort.InferenceSession(model_path)
        return session
    except Exception as e:
        st.error(f"❌ Erreur chargement: {e}")
        return None

def letterbox(image, new_shape=640, color=(114, 114, 114)):
    """Redimensionne l'image avec padding"""
    shape = image.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
    # Redimensionnement
    img = image.resize(new_unpad, Image.Resampling.LANCZOS)
    
    # Padding
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = dw // 2, dh // 2
    
    new_img = Image.new('RGB', (new_shape, new_shape), color)
    new_img.paste(img, (dw, dh))
    
    return new_img, (dw, dh, r)

def preprocess_image(image_pil, input_size=640):
    """Prépare l'image pour le modèle ONNX"""
    # Redimensionnement avec padding
    img_padded, (dw, dh, scale) = letterbox(image_pil, input_size)
    
    # Conversion en numpy et normalisation
    img_np = np.array(img_padded).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
    img_np = np.expand_dims(img_np, axis=0)  # CHW -> BCHW
    
    return img_np, (dw, dh, scale)

def postprocess_output(output, dw, dh, scale, conf_threshold=0.25):
    """Décodage de la sortie YOLOv8 ONNX"""
    # Format: [batch, 84, 8400] pour YOLOv8
    predictions = output[0]  # (84, 8400)
    
    boxes = []
    scores = []
    class_ids = []
    
    # Transposer pour avoir (8400, 84)
    predictions = predictions.T  # (8400, 84)
    
    for pred in predictions:
        conf = pred[4:]  # scores des classes
        class_id = np.argmax(conf)
        score = conf[class_id]
        
        if score >= conf_threshold:
            # Coordonnées relatives (cx, cy, w, h)
            cx, cy, w, h = pred[:4]
            
            # Conversion en coordonnées absolues dans l'image redimensionnée
            x1 = (cx - w/2) * 640
            y1 = (cy - h/2) * 640
            x2 = (cx + w/2) * 640
            y2 = (cy + h/2) * 640
            
            # Ajuster pour le padding
            x1 = (x1 - dw) / scale
            y1 = (y1 - dh) / scale
            x2 = (x2 - dw) / scale
            y2 = (y2 - dh) / scale
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
            class_ids.append(int(class_id))
    
    # NMS simple (non-max suppression)
    if len(boxes) > 0:
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Tri par score décroissant
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            
            if len(indices) == 1:
                break
            
            # Calcul IoU
            box_i = boxes[i]
            rest_boxes = boxes[indices[1:]]
            
            x1 = np.maximum(box_i[0], rest_boxes[:, 0])
            y1 = np.maximum(box_i[1], rest_boxes[:, 1])
            x2 = np.minimum(box_i[2], rest_boxes[:, 2])
            y2 = np.minimum(box_i[3], rest_boxes[:, 3])
            
            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)
            inter = w * h
            
            area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
            area_rest = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
            
            iou = inter / (area_i + area_rest - inter)
            
            # Garder les boxes avec IoU < seuil
            indices = indices[1:][iou < 0.45]
        
        return boxes[keep], scores[keep], class_ids[keep]
    
    return [], [], []

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    conf_threshold = st.slider("Seuil de confiance", 0.10, 0.90, 0.25, 0.05)
    box_thickness = st.slider("Épaisseur contours", 2, 12, 6)
    font_scale = st.slider("Taille texte", 0.8, 2.0, 1.4, 0.1)

st.title("🛒 Shelf Recognition — Stage 1")
st.markdown("Détection multi-objets — **YOLOv8s (ONNX)**")

model = load_model()
if model is None:
    st.stop()

st.success("✅ Modèle ONNX chargé")
st.divider()

# Upload
uploaded = st.file_uploader("Uploadez une photo de rayon", type=["jpg", "jpeg", "png"])

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")
    orig_img = img_pil.copy()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image originale")
        st.image(orig_img, use_container_width=True)

    with st.spinner("🔍 Détection..."):
        t0 = time.time()
        
        # Prétraitement
        img_input, (dw, dh, scale) = preprocess_image(orig_img)
        
        # Inférence ONNX
        inputs = {model.get_inputs()[0].name: img_input}
        outputs = model.run(None, inputs)
        
        # Post-traitement
        boxes, scores, class_ids = postprocess_output(outputs, dw, dh, scale, conf_threshold)
        
        latency_ms = (time.time() - t0) * 1000

    # Dessiner les résultats
    img_result = orig_img.copy()
    draw = ImageDraw.Draw(img_result)
    
    detections = []
    class_counts = {}
    
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"classe_{cls_id}"
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))
        
        # Rectangle épais
        for i in range(box_thickness):
            draw.rectangle([(x1+i, y1+i), (x2-i, y2-i)], outline=color)
        
        # Texte
        label = f"{cls_name} {score:.2f}"
        font_size = int(28 * font_scale)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Fond texte
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.rectangle([(x1, y1 - text_h - 12), (x1 + text_w + 12, y1)], fill=color)
        draw.text((x1 + 6, y1 - text_h - 6), label, fill=(255, 255, 255), font=font)
        
        detections.append({'classe': cls_name, 'confiance': round(score, 3)})
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    with col2:
        st.subheader(f"Résultat — {len(detections)} détection(s)")
        st.image(img_result, use_container_width=True)

    # Métriques
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Objets détectés", len(detections))
    c2.metric("📦 Classes présentes", len(class_counts))
    c3.metric("⚡ Latence", f"{latency_ms:.1f} ms")

    if class_counts:
        st.divider()
        st.subheader("Récapitulatif")
        df = pd.DataFrame([
            {'Classe': cls, 'Instances': cnt}
            for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])
        ])
        st.dataframe(df, hide_index=True, use_container_width=True)

    # Téléchargement
    st.divider()
    buf = io.BytesIO()
    img_result.save(buf, format="PNG")
    st.download_button("💾 Télécharger", data=buf.getvalue(), file_name="result.png", mime="image/png")