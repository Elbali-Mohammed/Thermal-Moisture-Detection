from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from inference import build_models, load_config, predict
from reporting import build_batch_inspection_report, build_inspection_report


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "ensemble_config.json"

st.set_page_config(page_title="Thermal Moisture Inspector", page_icon="💧", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at 12% 4%, #173e45 0%, #0c2027 30%, #07151b 75%);
            color: #ecf8f5;
        }
        .block-container { max-width: 1250px; padding-top: 2.2rem; padding-bottom: 5rem; }
        h1, h2, h3 { color: #f2fcfa !important; }
        .hero {
            padding: 2rem 2.25rem;
            border: 1px solid rgba(110, 225, 194, .25);
            border-radius: 22px;
            background: linear-gradient(120deg, rgba(30, 100, 101, .55), rgba(13, 37, 47, .82));
            box-shadow: 0 18px 45px rgba(0, 0, 0, .18);
            margin-bottom: 1.7rem;
        }
        .hero h1 { margin: 0; font-size: 2.3rem; }
        .hero p { margin: .55rem 0 0; color: #c8e4df; font-size: 1.05rem; }
        .section-label {
            color: #83e1c4; font-weight: 700; letter-spacing: .09em;
            text-transform: uppercase; font-size: .76rem;
        }
        [data-testid="stFileUploader"] {
            padding: 1.1rem; border: 1px dashed #62cbb2; border-radius: 16px;
            background: rgba(30, 89, 85, .18);
        }
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #16766f, #1ba482);
            padding: 1.1rem 1.35rem; border-radius: 16px;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: white !important; }
        .collaboration-card {
            margin-top: 2rem; padding: 1.2rem 1.4rem; border-radius: 16px;
            border-left: 4px solid #69d8bb; background: rgba(255, 255, 255, .07);
            color: #d5ece8;
        }
        .footer {
            position: fixed; left: 1.25rem; bottom: .75rem; z-index: 1000;
            color: #a5c8c2; font-size: .78rem;
        }
        .footer a { color: #83e1c4; text-decoration: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the final ensemble…")
def load_ensemble():
    config = load_config(CONFIG_PATH)
    # Resolve relative checkpoint paths from this config file, not the launch folder.
    config["checkpoints"] = {
        name: str((CONFIG_PATH.parent / path).resolve())
        for name, path in config["checkpoints"].items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return build_models(config, device), config, device


def resize_for_display(array: np.ndarray, size: tuple[int, int], is_mask: bool = False) -> np.ndarray:
    mode = "L" if array.ndim == 2 else "RGB"
    image = Image.fromarray(array.astype(np.uint8), mode=mode)
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    return np.asarray(image.resize(size, resample))


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    humid = mask.astype(bool)
    overlay[humid] = 0.55 * overlay[humid] + 0.45 * np.array([0, 220, 90])
    return overlay.astype(np.uint8)


st.markdown(
    """
    <div class="hero">
        <div class="section-label">Thermal image analysis</div>
        <h1>Thermal Moisture Inspector</h1>
        <p>Upload a thermal image to visualize potential moisture areas.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("This is a research prototype trained on a specific thermal-image dataset.")

inspector_notes = st.text_area(
    "Inspection notes (optional)",
    placeholder="Add observations, location details, or recommended follow-up actions...",
)
uploaded_files = st.file_uploader(
    "Upload one or more thermal images",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.markdown(
        """
        <div class="collaboration-card">
            <strong>Help develop the project further.</strong><br>
            Access to a high-quality, diverse thermal-image dataset would be greatly appreciated to support future model development and validation.
            For further development or collaboration, contact
            <a href="mailto:mohammed.elbali.officiel@gmail.com">mohammed.elbali.officiel@gmail.com</a>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='footer'>© ELBALI mohammed &nbsp;·&nbsp; <a href='mailto:mohammed.elbali.officiel@gmail.com'>Collaborate</a></div>",
        unsafe_allow_html=True,
    )
    st.stop()

try:
    models, config, device = load_ensemble()
except FileNotFoundError as error:
    st.error("A required checkpoint could not be found. Update its path in `app/ensemble_config.json`.")
    st.code(str(error))
    st.stop()
except RuntimeError as error:
    st.error("The checkpoint and model definition do not match. Check the architecture and checkpoint format notes in README.md.")
    st.code(str(error))
    st.stop()

with st.spinner(f"Inspecting {len(uploaded_files)} thermal image(s)…"):
    inspections = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        probability, mask = predict(image, models, config, device)
        display_image = np.asarray(image)
        display_size = image.size
        probability_display = np.asarray(
            Image.fromarray(probability.astype(np.float32), mode="F").resize(display_size, Image.Resampling.BILINEAR)
        )
        mask_display = resize_for_display(mask * 255, display_size, is_mask=True) > 0
        overlay = make_overlay(display_image, mask_display)
        humid_area = 100 * float(mask.mean())
        inspections.append({
            "filename": uploaded_file.name,
            "image": image,
            "display_image": display_image,
            "probability": probability,
            "probability_display": probability_display,
            "mask": mask,
            "mask_display": mask_display,
            "overlay": overlay,
            "humid_area": humid_area,
        })

if len(inspections) > 1:
    st.subheader("Batch inspection summary")
    average_area = sum(item["humid_area"] for item in inspections) / len(inspections)
    highest_item = max(inspections, key=lambda item: item["humid_area"])
    summary_metric, average_metric, count_metric = st.columns(3)
    summary_metric.metric("Images inspected", len(inspections))
    average_metric.metric("Average humid area", f"{average_area:.1f}%")
    count_metric.metric("Highest humid area", f"{highest_item['humid_area']:.1f}%")
    st.caption(f"Priority image: {highest_item['filename']}")
    st.dataframe(
        [{"Image": item["filename"], "Detected humid area": f"{item['humid_area']:.1f}%"} for item in inspections],
        use_container_width=True,
        hide_index=True,
    )
    batch_report = build_batch_inspection_report(inspections=inspections, notes=inspector_notes)
    st.download_button(
        "Download Complete Batch Inspection Report",
        data=batch_report,
        file_name="thermal_moisture_batch_inspection_report.pdf",
        mime="application/pdf",
        key="batch_report",
    )

for index, item in enumerate(inspections):
    if len(inspections) > 1:
        st.divider()
        st.subheader(item["filename"])

    metric, details = st.columns([1, 3])
    metric.metric("Detected humid area", f"{item['humid_area']:.1f}%")
    details.caption("Estimated proportion of the image identified as a potential moisture area.")

    col1, col2 = st.columns(2)
    col1.image(item["display_image"], caption="Original thermal image", use_container_width=True)
    col2.image(item["overlay"], caption="Moisture inspection overlay", use_container_width=True)

    with st.expander("Show Advanced Model Diagnostic Outputs"):
        diagnostic_left, diagnostic_right = st.columns(2)
        diagnostic_left.image(item["mask_display"].astype(np.uint8) * 255, caption="Binary humidity mask", clamp=True, use_container_width=True)
        diagnostic_right.image(item["probability_display"], caption="Ensemble humidity probability", clamp=True, use_container_width=True)

    if len(inspections) == 1:
        report = build_inspection_report(
            original=item["image"],
            overlay=item["overlay"],
            mask=item["mask"],
            probability=item["probability"],
            filename=item["filename"],
            humid_area=item["humid_area"],
            notes=inspector_notes,
        )
        report_name = f"inspection_report_{Path(item['filename']).stem}.pdf"
        st.download_button(
            "Download Inspection Report",
            data=report,
            file_name=report_name,
            mime="application/pdf",
            key=f"report_{index}_{item['filename']}",
            use_container_width=False,
        )

st.markdown(
    """
    <div class="collaboration-card">
        <strong>Further development &amp; collaboration</strong><br>
        Access to a high-quality, diverse thermal-image dataset would be greatly appreciated to support future model development and validation.
        For collaboration, contact
        <a href="mailto:mohammed.elbali.officiel@gmail.com">mohammed.elbali.officiel@gmail.com</a>.
    </div>
    <div class="footer">© ELBALI mohammed &nbsp;·&nbsp; <a href="mailto:mohammed.elbali.officiel@gmail.com">Collaborate</a></div>
    """,
    unsafe_allow_html=True,
)
