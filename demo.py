import gradio as gr
import supervision as sv
import numpy as np
import tempfile, os
from PIL import Image

# =====================================================================
# BẠN CHỈ CẦN THAY ĐỔI DẤU '#' Ở 2 DÒNG DƯỚI ĐÂY ĐỂ CHỌN MÔ HÌNH:

# MODEL_TYPE = "YOLO"        # Bỏ comment dòng này để dùng YOLO cục bộ (best.pt)
MODEL_TYPE = "RF-DETR"  # Bỏ comment dòng này để dùng API Roboflow (RF-DETR)

# =====================================================================

print(f"[*] Đang khởi tạo mô hình: {MODEL_TYPE} ...")

if MODEL_TYPE == "YOLO":
    # Load YOLO cục bộ
    from ultralytics import YOLO
    model = YOLO("best.pt") 
else:
    # Load Roboflow API
    from roboflow import Roboflow
    rf = Roboflow(api_key="tIkFrVTvasBP2xDROTv7")
    model = (
        rf.workspace("dasha-33ndt")
          .project("helmet-wearing-detection-vweez-5c0da")
          .version(1)
          .model
    )

# --- TÙY CHỈNH NÉT VẼ VÀ CHỮ NHỎ LẠI ---
# thickness=1 (Nét mỏng), text_scale=0.4 (Chữ nhỏ gọn), text_padding=5 (Viền chữ nhỏ)
box_annotator   = sv.BoxAnnotator(thickness=1)
label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1, text_padding=5)

def predict(image):
    # --- PHẦN 1: CHẠY SUY LUẬN TÙY THEO MÔ HÌNH ---
    if MODEL_TYPE == "YOLO":
        # Cách YOLO chạy
        results = model.predict(image, conf=0.4)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        class_names = [model.names[class_id] for class_id in detections.class_id]
        confidences = detections.confidence.tolist()
        
        json_predictions = [{"class": c, "confidence": round(float(conf), 4)} for c, conf in zip(class_names, confidences)]
        
    elif MODEL_TYPE == "RF-DETR":
        # Cách Roboflow chạy
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            image.save(f.name)
            tmp_path = f.name
        try:
            result = model.predict(tmp_path, confidence=40).json()
        finally:
            os.unlink(tmp_path)
            
        detections = sv.Detections.from_inference(result)
        class_names = [p["class"] for p in result["predictions"]]
        json_predictions = result["predictions"]

    # --- PHẦN 2: VẼ LÊN ẢNH ---
    # Bỏ phần % độ tin cậy đi (giống conf=False) để nhãn gọn gàng nhất có thể
    labels = class_names

    img = np.array(image)
    img = box_annotator.annotate(img, detections=detections)
    img = label_annotator.annotate(img, detections=detections, labels=labels)

    # --- PHẦN 3: THỐNG KÊ ---
    # Gom chung "without_helmet" (của YOLO) và "no-helmet" (của RF) để thống kê không bị lỗi
    helmet_count      = class_names.count("helmet")
    no_helmet_count   = class_names.count("without_helmet") + class_names.count("no-helmet")
    two_wheeler_count = class_names.count("two_wheeler")
    total             = len(detections)

    stats_html = f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-top:8px;">
      <div style="flex:1; min-width:100px; background:var(--color-background-secondary); border-radius:8px; padding:14px 16px; text-align:center;">
        <div style="font-size:13px; color:var(--color-text-secondary); margin-bottom:4px;">Total detected</div>
        <div style="font-size:26px; font-weight:500; color:var(--color-text-primary);">{total}</div>
      </div>
      <div style="flex:1; min-width:100px; background:var(--color-background-secondary); border-radius:8px; padding:14px 16px; text-align:center;">
        <div style="font-size:13px; color:var(--color-text-secondary); margin-bottom:4px;">🏍️ Two Wheeler</div>
        <div style="font-size:26px; font-weight:500; color:#3B82F6;">{two_wheeler_count}</div>
      </div>
      <div style="flex:1; min-width:100px; background:var(--color-background-secondary); border-radius:8px; padding:14px 16px; text-align:center;">
        <div style="font-size:13px; color:var(--color-text-secondary); margin-bottom:4px;">✅ With helmet</div>
        <div style="font-size:26px; font-weight:500; color:#1D9E75;">{helmet_count}</div>
      </div>
      <div style="flex:1; min-width:100px; background:var(--color-background-secondary); border-radius:8px; padding:14px 16px; text-align:center;">
        <div style="font-size:13px; color:var(--color-text-secondary); margin-bottom:4px;">⚠️ No helmet</div>
        <div style="font-size:26px; font-weight:500; color:#D85A30;">{no_helmet_count}</div>
      </div>
    </div>
    """

    return Image.fromarray(img), stats_html, json_predictions

css = """
.gradio-container { max-width: 960px !important; margin: auto; }
#title { text-align: center; margin-bottom: 0.25rem; }
#model-indicator { text-align: center; color: #1D9E75; font-weight: bold; font-size: 1.1rem; margin-bottom: 0.5rem; }
#subtitle { text-align: center; color: #888; font-size: 0.95rem; margin-bottom: 1.5rem; }
#run-btn { background: #1D9E75 !important; color: white !important; border: none !important; font-size: 1rem; }
#run-btn:hover { background: #0F6E56 !important; }
"""

with gr.Blocks(css=css, title="Helmet Detection") as demo:
    gr.Markdown("# 🪖 Helmet Wearing Detection", elem_id="title")
    
    # --- HIỂN THỊ TRẠNG THÁI MÔ HÌNH LÊN APP ---
    gr.Markdown(f"<div id='model-indicator'>🚀 Active Model: {MODEL_TYPE}</div>")
    
    gr.Markdown("Upload a workplace image to detect who is wearing a helmet.", elem_id="subtitle")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            inp_image = gr.Image(type="pil", label="Input image", height=340)
            btn = gr.Button("Run detection", variant="primary", elem_id="run-btn")

        with gr.Column(scale=1):
            out_image = gr.Image(label="Annotated result", height=340)

    out_stats = gr.HTML(label="Summary")

    with gr.Accordion("Raw predictions (JSON)", open=False):
        out_json = gr.JSON()

    btn.click(
        predict,
        inputs=[inp_image],
        outputs=[out_image, out_stats, out_json]
    )

if __name__ == "__main__":
    demo.launch(share=True)