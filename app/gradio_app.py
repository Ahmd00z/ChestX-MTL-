#!/usr/bin/env python3
"""
ChestX-MTL Gradio Web Interface
Beautiful interactive UI for chest X-ray analysis.

Run: python app/gradio_app.py
"""
import os
import sys
import numpy as np
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.inference import ChestXInference


# Load model
MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/checkpoints/best_model.pth")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config/config.yaml")

print("Loading model...")
try:
    engine = ChestXInference(
        checkpoint_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        device="auto"
    )
    print("Model loaded!")
except Exception as e:
    print(f"Error loading model: {e}")
    engine = None


def analyze_xray(image, cls_threshold, seg_threshold):
    """Analyze X-ray and return results."""
    if engine is None:
        return None, "Error: Model not loaded", ""

    if image is None:
        return None, "Please upload an image", ""

    # Run inference
    result = engine.predict(
        image,
        cls_threshold=cls_threshold,
        seg_threshold=seg_threshold
    )

    # Generate visualization
    vis_image = engine.visualize(image, result)

    # Build text report
    diseases = result["classification"]["diseases"]
    detected = [d for d in diseases if d["detected"]]

    report = "## 📋 Analysis Report\n\n"

    if detected:
        report += "### ⚠️ Detected Abnormalities\n\n"
        for d in detected:
            report += f"- **{d['disease']}**: `{d['probability']*100:.1f}%`\n"
    else:
        report += "### ✅ No Abnormalities Detected\n\n"
        report += "The model did not detect any abnormalities above the threshold.\n"

    report += f"\n### 📊 Statistics\n"
    report += f"- Affected Area: `{result['segmentation']['affected_area_ratio']*100:.2f}%`\n"
    report += f"- Classification Threshold: `{cls_threshold}`\n"
    report += f"- Segmentation Threshold: `{seg_threshold}`\n"

    # All probabilities table
    report += "\n### 📈 All Probabilities\n\n"
    report += "| Disease | Probability | Status |\n"
    report += "|---------|-------------|--------|\n"
    for d in diseases:
        status = "🔴 Detected" if d["detected"] else "🟢 Normal"
        report += f"| {d['disease']} | {d['probability']*100:.1f}% | {status} |\n"

    return vis_image, report, "Analysis complete!"


# Custom CSS for beautiful UI
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif !important;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 20px;
}
.header h1 {
    margin: 0;
    font-size: 2.5em;
    font-weight: 700;
}
.header p {
    margin: 10px 0 0 0;
    opacity: 0.9;
    font-size: 1.1em;
}
.result-box {
    border-radius: 12px;
    padding: 15px;
    background: #f8f9fa;
}
"""

# Build Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div class="header">
        <h1>🫁 ChestX-MTL</h1>
        <p>Advanced Multi-Task AI for Chest X-Ray Analysis</p>
        <p style="font-size: 0.9em; opacity: 0.8;">
            Classification • Detection • Segmentation
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload X-Ray Image")
            input_image = gr.Image(
                type="pil",
                label="Chest X-Ray",
                height=400
            )

            with gr.Row():
                cls_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Classification Threshold"
                )
                seg_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Segmentation Threshold"
                )

            analyze_btn = gr.Button(
                "🔍 Analyze X-Ray",
                variant="primary",
                size="lg"
            )

            gr.Markdown("""
            ### ℹ️ Supported Findings
            Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, 
            Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, 
            Pleural Thickening, Pneumonia, Pneumothorax
            """)

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Analysis Results")

            with gr.Row():
                output_image = gr.Image(
                    label="Visualization",
                    height=400
                )

            output_report = gr.Markdown(
                label="Detailed Report",
                value="Upload an image and click Analyze to see results."
            )

            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                value="Ready"
            )

    # Examples
    gr.Examples(
        examples=[],
        inputs=input_image,
        label="Example X-Rays (upload your own)"
    )

    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666; padding: 10px;">
        <p>Built with PyTorch • FastAPI • Gradio</p>
        <p>ChestX-MTL v2.0.0</p>
    </div>
    """)

    # Event handlers
    analyze_btn.click(
        fn=analyze_xray,
        inputs=[input_image, cls_slider, seg_slider],
        outputs=[output_image, output_report, status_text]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
