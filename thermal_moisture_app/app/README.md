# Thermal Moisture Segmentation Prototype

This Streamlit application provides a local inference interface for the final frozen ensemble developed in the project.

The system takes thermal image(s) as input and performs pixel-level segmentation of suspected moisture regions using three complementary deep-learning models:

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

## 2. Requirements

-Python 3.x
-Git LFS
-Dependencies listed in app/requirements.txt

Because the trained checkpoints are stored using Git LFS, Git LFS must be installed before cloning or downloading the repository.


## 3. Install and run

From the thermal_moisture_app directory:

```powershell
python -m pip install -r app/requirements.txt
streamlit run app/app.py
```

Streamlit will open the local application in a browser. 
Upload one or more thermal images to run the ensemble inference pipeline.
