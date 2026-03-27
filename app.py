import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
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

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    if not os.path.exists(model_path):
        st.error(f"❌ Modèle introuvable: {model_path}")
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        return None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    conf_threshold = st.slider("Seuil de confiance", 0.10, 0.90, 0.25, 0.05)
    box_thickness = st.slider("Épaisseur contours", 2, 12, 6)
    font_scale = st.slider("Taille texte", 0.8, 2.0, 1.4, 0.1)

st.title("🛒 Shelf Recognition — Stage 1")
st.markdown("Détection multi-objets — **YOLOv8s**")

model = load_model()
if model is None:
    st.stop()

st.success("✅ Modèle chargé")
st.divider()

# Upload
uploaded = st.file_uploader("Uploadez une photo de rayon", type=["jpg", "jpeg", "png"])

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")
    img_np = np.array(img_pil)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image originale")
        st.image(img_pil, use_container_width=True)

    with st.spinner("🔍 Détection..."):
        t0 = time.time()
        results = model.predict(img_np, conf=conf_threshold, verbose=False)
        latency_ms = (time.time() - t0) * 1000

    # Dessiner avec PIL (pas d'OpenCV)
    img_result = img_pil.copy()
    draw = ImageDraw.Draw(img_result)
    
    detections = []
    class_counts = {}

    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                color = CLASS_COLORS.get(cls_name, (255, 255, 255))
                
                # Rectangle épais
                for i in range(box_thickness):
                    draw.rectangle([(x1+i, y1+i), (x2-i, y2-i)], outline=color)
                
                # Texte
                label = f"{cls_name} {conf:.2f}"
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
                
                detections.append({'classe': cls_name, 'confiance': round(conf, 3)})
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