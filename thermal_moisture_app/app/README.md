# Thermal Moisture Segmentation Prototype

This Streamlit app packages the **final frozen ensemble** from the project:

`0.30 × DeepLabV3+ ResNet-50 + 0.30 × SegFormer B0 + 0.40 × SegFormer B1`

It applies the final decision threshold: `probability >= 0.625`.

## 1. Project Structure

The three final checkpoints are included in the repository under `models/` and are managed using Git LFS.

```text
thermal_moisture_app/
├── app/
│   ├── app.py
│   ├── inference.py
│   ├── reporting.py
│   ├── ensemble_config.json
│   └── requirements.txt
└── models/
    ├── deeplabv3plus_resnet50_best.pth
    ├── segformer_mit_b0_best.pth
    └── segformer_mit_b1_best.pth
```

If the model filenames or locations are changed, update the corresponding three paths under `checkpoints` in `app/ensemble_config.json`.


## 2. Install and run

From the thermal_moisture_app directory:

```powershell
python -m pip install -r app/requirements.txt
streamlit run app/app.py
```

Streamlit will open the local application in a browser. Upload a PNG, JPG, JPEG, TIFF, or TIF thermal image.

then make the matching architecture-only adjustment (for example, `classes`, `num_labels`, or checkpoint key). This does not change the finalized model or evaluation.
