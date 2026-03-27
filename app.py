import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import os
import time
import io
import pandas as pd
import matplotlib.pyplot as plt

# Désactiver l'import OpenCV dans ultralytics
os.environ["ULTRALYTICS_NO_OPENCV"] = "1"

# ─── Configuration page ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shelf Recognition — Stage 1",
    page_icon="🛒",
    layout="wide"
)

# ─── Couleurs par classe (RGB) ────────────────────────────────────────────────
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

# ─── Chargement modèle (cache) ────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# ─── Prédiction avec PIL ──────────────────────────────────────────────────────
def predict_frame_pil(model, image_pil, conf_threshold, box_thickness=6, font_scale=1.4):
    results = model.predict(image_pil, conf=conf_threshold, verbose=False)
    result = results[0]
    
    img_draw = image_pil.copy()
    draw = ImageDraw.Draw(img_draw)
    
    detections = []
    class_counts = {}

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu())
            conf_val = float(box.conf[0].cpu())
            cls_name = result.names[cls_id]
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            
            # Rectangle
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=box_thickness)
            
            # Texte
            label = f"{cls_name} {conf_val:.2f}"
            font_size = max(20, int(24 * font_scale))
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Calcul taille texte
            bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Fond
            draw.rectangle([(x1, y1 - text_height - 10), (x1 + text_width + 10, y1)], fill=color)
            draw.text((x1 + 5, y1 - text_height - 5), label, fill=(255, 255, 255), font=font)
            
            detections.append({'classe': cls_name, 'confiance': round(conf_val, 3)})
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    return img_draw, detections, class_counts

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")

    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    conf_threshold = st.slider("Seuil de confiance", 0.10, 0.90, 0.25, 0.05)
    box_thickness = st.slider("Épaisseur des contours", 2, 12, 6)
    font_scale = st.slider("Taille du texte", 0.8, 2.0, 1.4, 0.1)

    st.divider()
    st.markdown("**Légende des classes**")
    for cls, color in CLASS_COLORS.items():
        hex_c = "#{:02x}{:02x}{:02x}".format(*color)
        st.markdown(
            f"<span style='background:{hex_c}; padding:2px 10px; "
            f"border-radius:4px; color:white; font-size:12px'>{cls}</span>",
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🛒 Shelf Recognition — Détection Stage 1")
st.markdown("Détection multi-objets par famille de produits — **YOLOv8s**")

# ─── Chargement modèle ────────────────────────────────────────────────────────
if not os.path.exists(model_path):
    st.error(f"❌ Modèle introuvable : `{model_path}`")
    st.info("💡 Placez votre fichier `best.pt` dans le dossier de l'application.")
    st.stop()

model = load_model(model_path)
st.success(f"✅ Modèle chargé — `{os.path.basename(model_path)}`")
st.divider()

# ─── Choix du mode ────────────────────────────────────────────────────────────
mode = st.radio(
    "Choisissez le mode d'analyse",
    ["📷 Upload image"],
    horizontal=True
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MODE — UPLOAD IMAGE
# ══════════════════════════════════════════════════════════════════════════════
uploaded = st.file_uploader(
    "Uploadez une photo de rayon (.jpg / .png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image originale")
        st.image(img_pil, use_container_width=True)

    with st.spinner("🔍 Détection en cours..."):
        t0 = time.time()
        img_result, detections, class_counts = predict_frame_pil(
            model, img_pil, conf_threshold, box_thickness, font_scale
        )
        latency_ms = (time.time() - t0) * 1000

    with col2:
        st.subheader(f"Résultat — {len(detections)} détection(s)")
        st.image(img_result, use_container_width=True)

    # Métriques
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Objets détectés", len(detections))
    c2.metric("📦 Classes présentes", len(class_counts))
    c3.metric("⚡ Latence", f"{latency_ms:.1f} ms")

    # Tableau + graphique
    if class_counts:
        st.divider()
        st.subheader("Récapitulatif par classe")
        col_a, col_b = st.columns([1, 1])

        rows = []
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            confs = [d['confiance'] for d in detections if d['classe'] == cls]
            rows.append({
                'Classe': cls,
                'Instances': cnt,
                'Conf. moy.': f"{np.mean(confs):.2f}",
                'Conf. max.': f"{max(confs):.2f}",
            })

        with col_a:
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)

        with col_b:
            fig, ax = plt.subplots(figsize=(5, max(2, len(rows) * 0.5)))
            classes_list = [r['Classe'] for r in rows]
            counts_list = [r['Instances'] for r in rows]
            colors_hex = [
                "#{:02x}{:02x}{:02x}".format(*CLASS_COLORS.get(c, (100, 100, 100)))
                for c in classes_list
            ]
            ax.barh(classes_list, counts_list, color=colors_hex, edgecolor='white')
            ax.set_xlabel("Instances")
            ax.set_title("Distribution des détections", fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Téléchargement
    st.divider()
    buf = io.BytesIO()
    img_result.save(buf, format="PNG")
    st.download_button(
        "💾 Télécharger le résultat",
        data=buf.getvalue(),
        file_name="detection_result.png",
        mime="image/png"
    )