import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import time
import io
import pandas as pd
import matplotlib.pyplot as plt

# ─── Configuration page ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shelf Recognition — Stage 1",
    page_icon="🛒",
    layout="wide"
)

# ─── Couleurs par classe ──────────────────────────────────────────────────────
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
def load_model(model_path):
    return YOLO(model_path)

def predict_frame(model, image_np, conf_threshold, box_thickness=6, font_scale=1.4):
    results = model.predict(image_np, conf=conf_threshold, verbose=False)
    result = results[0]
    img_draw = image_np.copy()
    detections = []
    class_counts = {}

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu())
            conf_val = float(box.conf[0].cpu())
            cls_name = result.names[cls_id]
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            bgr = (color[2], color[1], color[0])

            # Rectangle épais
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), bgr, box_thickness)
            
            # Texte
            label = f"{cls_name} {conf_val:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
            
            # Fond texte
            cv2.rectangle(img_draw, (x1, y1 - th - 15), (x1 + tw + 15, y1), bgr, -1)
            cv2.putText(img_draw, label, (x1 + 8, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)
            
            detections.append({'classe': cls_name, 'confiance': round(conf_val, 3)})
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    return img_draw, detections, class_counts

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    conf_threshold = st.slider("Seuil de confiance", 0.10, 0.90, 0.25, 0.05)
    box_thickness = st.slider("Épaisseur contours", 2, 12, 6)
    font_scale = st.slider("Taille texte", 0.8, 2.0, 1.4, 0.1)

    st.divider()
    st.markdown("**Légende des classes**")
    for cls, color in CLASS_COLORS.items():
        hex_c = "#{:02x}{:02x}{:02x}".format(*color)
        st.markdown(f"<span style='background:{hex_c}; padding:2px 10px; border-radius:4px; color:white'>{cls}</span>", unsafe_allow_html=True)

# ─── Main ────────────────────────────────────────────────────────────────────
st.title("🛒 Shelf Recognition — Stage 1")
st.markdown("Détection multi-objets par famille de produits — **YOLOv8s**")

if not os.path.exists(model_path):
    st.error(f"❌ Modèle introuvable : `{model_path}`")
    st.stop()

model = load_model(model_path)
st.success(f"✅ Modèle chargé — `{os.path.basename(model_path)}`")
st.divider()

# ─── Upload ───────────────────────────────────────────────────────────────────
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
        img_result, detections, class_counts = predict_frame(
            model, img_np, conf_threshold, box_thickness, font_scale
        )
        latency_ms = (time.time() - t0) * 1000

    with col2:
        st.subheader(f"Résultat — {len(detections)} détection(s)")
        st.image(img_result, use_container_width=True)

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
    Image.fromarray(img_result).save(buf, format="PNG")
    st.download_button("💾 Télécharger", data=buf.getvalue(), file_name="result.png", mime="image/png")