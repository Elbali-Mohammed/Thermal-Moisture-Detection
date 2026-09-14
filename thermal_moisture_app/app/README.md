# Thermal Moisture Segmentation Prototype

This Streamlit app packages the **final frozen ensemble** from the project:

`0.30 × DeepLabV3+ ResNet-50 + 0.30 × SegFormer B0 + 0.40 × SegFormer B1`

It applies the final decision threshold: `probability >= 0.625`.

## 1. Place the three final checkpoints

Create a `models` folder beside `app`, then place the exact checkpoints used for the final test evaluation there:

```text
project-root/
├── app/
│   ├── app.py
│   ├── inference.py
│   └── ensemble_config.json
└── models/
    ├── deeplabv3plus_resnet50_best.pth
    ├── segformer_mit_b0_best.pth
    └── segformer_mit_b1_best.pth
```

If your filenames or location are different, edit only the three paths under `checkpoints` in `app/ensemble_config.json`.


## 2. Install and run

From the project root:

```powershell
python -m pip install -r app/requirements.txt
streamlit run app/app.py
```

Streamlit opens the local application in a browser. Upload a PNG, JPG, JPEG, TIFF, or TIF thermal image.

## Checkpoint compatibility

`inference.py` supports a raw PyTorch state dictionary and common wrappers named `model_state_dict`, `state_dict`, or `model`.

If you previously installed the newest `transformers` package and see a long
`Missing key(s)` / `Unexpected key(s)` SegFormer error, run this once before
starting the app:

```powershell
python -m pip uninstall -y transformers
python -m pip install "transformers>=4.38,<5.0"
```

Then start Streamlit again. Your checkpoints use the Transformers 4.x internal
parameter names; this is a package compatibility issue, not a damaged model.

It recreates the architectures assumed by the ensemble notebook:

- `segmentation_models_pytorch.DeepLabV3Plus`, ResNet-50 encoder, one output logit.
- Hugging Face `SegformerForSemanticSegmentation`, `nvidia/mit-b0` and `nvidia/mit-b1`, two output classes where class 1 is humidity.

If a checkpoint fails to load, do not retrain. Compare the cell where that particular model was created in the notebook with `build_models()` in `inference.py`, then make the matching architecture-only adjustment (for example, `classes`, `num_labels`, or checkpoint key). This does not change the finalized model or evaluation.
