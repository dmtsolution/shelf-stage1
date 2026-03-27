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

def letterbox(image_pil, new_shape=640, color=(114, 114, 114)):
    width, height = image_pil.size
    r = min(new_shape / height, new_shape / width)
    new_w = int(round(width * r))
    new_h = int(round(height * r))
    
    img = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    dw = (new_shape - new_w) // 2
    dh = (new_shape - new_h) // 2
    
    new_img = Image.new('RGB', (new_shape, new_shape), color)
    new_img.paste(img, (dw, dh))
    
    return new_img, (dw, dh, r, new_w, new_h)

def preprocess_image(image_pil, input_size=640):
    img_padded, (dw, dh, scale, new_w, new_h) = letterbox(image_pil, input_size)
    img_np = np.array(img_padded).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np, (dw, dh, scale, new_w, new_h, img_padded.size[0], img_padded.size[1])

def postprocess_output(output, dw, dh, scale, orig_w, orig_h, conf_threshold=0.25, iou_threshold=0.45):
    """Décodage correct pour YOLOv8 ONNX"""
    predictions = np.squeeze(output[0])  # (84, 8400)
    
    boxes = []
    scores = []
    class_ids = []
    
    # Transposer pour avoir (8400, 84)
    predictions = predictions.T
    
    for pred in predictions:
        # Les 4 premiers sont x_center, y_center, width, height (normalisés)
        cx, cy, w, h = pred[:4]
        
        # Les 80 suivants sont les scores de classe
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]
        
        if score >= conf_threshold:
            # Convertir les coordonnées normalisées (0-1) en pixels dans l'image redimensionnée (640x640)
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
            class_ids.append(class_id)
    
    if len(boxes) == 0:
        return [], [], []
    
    # NMS (Non-Max Suppression)
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)
    
    # Trier par score
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        
        if len(indices) == 1:
            break
        
        box_i = boxes[i]
        rest_indices = indices[1:]
        rest_boxes = boxes[rest_indices]
        
        # Calculer IoU
        x1 = np.maximum(box_i[0], rest_boxes[:, 0])
        y1 = np.maximum(box_i[1], rest_boxes[:, 1])
        x2 = np.minimum(box_i[2], rest_boxes[:, 2])
        y2 = np.minimum(box_i[3], rest_boxes[:, 3])
        
        inter_w = np.maximum(0, x2 - x1)
        inter_h = np.maximum(0, y2 - y1)
        inter = inter_w * inter_h
        
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        area_rest = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
        
        iou = inter / (area_i + area_rest - inter + 1e-6)
        
        indices = rest_indices[iou < iou_threshold]
    
    return boxes[keep], scores[keep], class_ids[keep]

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    conf_threshold = st.slider("Seuil de confiance", 0.10, 0.90, 0.25, 0.05)
    box_thickness = st.slider("Épaisseur contours", 2, 12, 6)
    font_scale = st.slider("Taille texte", 0.8, 2.0, 1.4, 0.1)
    
    st.divider()
    st.markdown("**📋 Légende des classes**")
    for cls, color in CLASS_COLORS.items():
        hex_c = "#{:02x}{:02x}{:02x}".format(*color)
        st.markdown(
            f"<span style='background:{hex_c}; padding:2px 10px; "
            f"border-radius:4px; color:white; font-size:12px'>{cls}</span>",
            unsafe_allow_html=True
        )

st.title("🛒 Shelf Recognition — Stage 1")
st.markdown("Détection multi-objets par famille de produits — **YOLOv8s (ONNX)**")

model = load_model()
if model is None:
    st.stop()

st.success("✅ Modèle ONNX chargé")
st.divider()

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
        
        img_input, (dw, dh, scale, new_w, new_h, padded_w, padded_h) = preprocess_image(orig_img)
        inputs = {model.get_inputs()[0].name: img_input}
        outputs = model.run(None, inputs)
        boxes, scores, class_ids = postprocess_output(
            outputs, dw, dh, scale, orig_img.size[0], orig_img.size[1], conf_threshold
        )
        
        latency_ms = (time.time() - t0) * 1000

    # Dessiner les résultats
    img_result = orig_img.copy()
    draw = ImageDraw.Draw(img_result)
    
    detections = []
    class_counts = {}
    
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        
        # S'assurer que les coordonnées sont dans l'image
        w, h = orig_img.size
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"classe_{cls_id}"
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))
        
        # Rectangle
        for i in range(box_thickness):
            draw.rectangle([(x1+i, y1+i), (x2-i, y2-i)], outline=color)
        
        # Texte
        label = f"{cls_name} {score:.2f}"
        font_size = int(28 * font_scale)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Position du texte (au-dessus de la box)
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        text_y = y1 - text_h - 5
        if text_y < 0:
            text_y = y1 + 5
        
        draw.rectangle([(x1, text_y), (x1 + text_w + 10, text_y + text_h + 5)], fill=color)
        draw.text((x1 + 5, text_y + 2), label, fill=(255, 255, 255), font=font)
        
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

    # Tableau récapitulatif
    if class_counts:
        st.divider()
        st.subheader("📊 Récapitulatif par classe")
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
    st.download_button("💾 Télécharger le résultat", data=buf.getvalue(), file_name="detection_result.png", mime="image/png")