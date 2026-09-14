# Thermal Moisture Detection



A deep-learning system for detecting and segmenting moisture/humidity regions in thermal images using semantic segmentation.



The project explores multiple deep-learning architectures, controlled model optimization, model comparison, and ensemble learning, followed by deployment as a local inference application.



## Overview



Moisture detection from thermal imagery can support the inspection of buildings and infrastructure by identifying regions with thermal patterns associated with humidity.



This project develops a computer-vision pipeline that takes a thermal image as input and produces a pixel-level segmentation of suspected moisture regions.



The final system combines three complementary segmentation models:



- **DeepLabV3+ with ResNet-50**

- **SegFormer MIT-B0**

- **SegFormer MIT-B1**



Their predictions are combined through a weighted ensemble to obtain the final moisture segmentation.



## Project Workflow



The project followed a progressive workflow, starting from the study of existing research and data preparation, followed by four main experimental stages:



### Preliminary Research and Data Preparation



1. Literature Review and Related Work

Study of existing research and previous approaches for thermal-image analysis, moisture detection, and semantic segmentation in order to identify relevant methodologies and establish the experimental direction.



2. Data Collection and Annotation

Collection of suitable thermal images from publicly available sources, followed by the preparation of pixel-level moisture segmentation masks for supervised learning.



### Experimental Development



3. Model Selection

Comparison of several semantic-segmentation architectures to identify promising candidate models for the task.



4. Final Model Comparison

Quantitative and qualitative comparison of the selected models trained during the model-selection stage, including analysis of their predictions and segmentation quality. This stage identified the strongest individual model and motivated its subsequent optimization.



5. ResNet-50 Optimization

Controlled experiments investigating learning rates, loss functions, augmentation, resolution, discriminative learning rates, boundary-aware losses, and prediction thresholds in an attempt to further improve the selected ResNet-50 model.



6. Ensembling

Analysis of model agreement and complementarity, followed by weighted ensemble optimization to combine the complementary strengths of the selected models.

## Dataset



The thermal images used in this project originate from publicly available thermal imagery collected through **Roboflow**.



The pixel-level moisture segmentation masks were **created and prepared independently for this project**, providing the annotations required for supervised semantic segmentation.



The resulting dataset contains thermal images paired with corresponding binary moisture masks.



The original dataset used during development is publicly available on Kaggle:



**[Humidity Thermal Dataset](https://www.kaggle.com/datasets/ebyeager/humidity-thermal)**



The dataset itself is **not included in this repository**.



## Dataset Structure



The training data follows an image-mask segmentation setup:



```text

images/

   image_001.jpg

   image_002.jpg

   ...



masks/

   image_001.png

   image_002.png

   ...

```



Each mask corresponds to its associated thermal image and identifies the pixels belonging to the moisture region.



## Final Models



### DeepLabV3+ — ResNet-50



A convolutional semantic-segmentation architecture using ResNet-50 as the encoder.



The original ResNet-50 model was retained for the final ensemble after controlled optimization experiments showed that the optimized variants did not improve the final test performance sufficiently.



### SegFormer — MIT-B0



A transformer-based semantic-segmentation architecture selected as one of the complementary models.



### SegFormer — MIT-B1



A larger SegFormer variant that achieved the strongest validation performance among the selected individual models.



## Model Optimization



The ResNet-50 model was investigated through a series of controlled experiments, including:



- Learning-rate variations

- BCE + Dice loss

- Focal + Dice loss

- Prediction-threshold optimization

- Offline segmentation augmentation

- Higher input resolution

- Discriminative learning rates

- Boundary-aware loss



The experiments demonstrated an important distinction between validation improvement and generalization to the unseen test set. The final ensemble therefore uses the original ResNet-50 checkpoint rather than the individually optimized ResNet-50 variant.



## Ensemble



The final system combines the three selected models using a weighted probability ensemble.



| Model | Weight |
|---|---:|
| DeepLabV3+ ResNet-50 | 0.30 |
| SegFormer MIT-B0 | 0.30 |
| SegFormer MIT-B1 | 0.40 |



The final segmentation threshold is:



```text

0.625

```



The ensemble was selected after evaluating both model agreement and prediction complementarity.



## Final Test Performance



The final ensemble achieved:



| Metric | Score |
|---|---:|
| Dice Score | **0.8504** |
| IoU | **0.7724** |
| Precision | **0.8668** |
| Recall | **0.8732** |



These results correspond to the held-out test set used during the project evaluation.



## Application



A local inference application is included in:



```text

thermal_moisture_app/

```



The application loads the three trained models and performs ensemble inference on a thermal image.



It contains:



```text

thermal_moisture_app/

├── app/

│   ├── app.py

│   ├── inference.py

│   ├── reporting.py

│   ├── ensemble_config.json

│   ├── requirements.txt

│   └── README.md

│

└── models/

   ├── deeplabv3plus_resnet50_best.pth

   ├── segformer_mit_b0_best.pth

   └── segformer_mit_b1_best.pth

```



The trained checkpoints are distributed using **Git LFS**.



## Notebooks



The complete experimental workflow is provided in the `Notebooks/` directory:



```text

Notebooks/

├── 01_model_selection.ipynb

├── 02_resnet50_optimization.ipynb

├── 03_final_comparison.ipynb

└── 04_ensembling.ipynb

```



The notebooks were developed and executed in **Kaggle**.



Their original Kaggle-specific dataset and output paths have intentionally been preserved. Consequently, the notebooks are provided as a record of the experimental workflow and may require path adaptation to run in another environment.



## Repository Structure



```text

Thermal-Moisture-Detection/

│

├── Notebooks/

│   ├── 01_model_selection.ipynb

│   ├── 02_resnet50_optimization.ipynb

│   ├── 03_final_comparison.ipynb

│   └── 04_ensembling.ipynb

│

├── thermal_moisture_app/

│   ├── app/

│   └── models/

│

├── .gitattributes

├── .gitignore

└── README.md

```



The original dataset, literature resources, and intermediate experimental results are intentionally excluded from the repository.



## Limitations



The current system was developed using a publicly available thermal-image dataset and was not trained or validated on a dedicated dataset acquired from a specific target building.



Therefore, the current model should be considered a **general thermal moisture-segmentation prototype**, rather than a validated deployment system for a specific building.



Performance may vary depending on:



- Thermal camera characteristics

- Environmental conditions

- Building materials

- Moisture patterns

- Image acquisition conditions

- Differences between the training data and real-world deployment environments



## Future Work



Potential extensions include:



- Acquisition of dedicated thermal imagery from real buildings

- RGB + thermal multimodal learning

- Integration of photogrammetry and 3D reconstruction

- Building-specific fine-tuning

- Quantitative moisture-level estimation

- Larger and more diverse datasets

- Domain adaptation for different thermal cameras and environments



## Technologies



- Python

- PyTorch

- Segmentation Models / Deep Learning

- DeepLabV3+

- SegFormer

- ResNet-50

- Computer Vision

- Semantic Segmentation

- Git LFS

- Kaggle



## License


This repository contains code, notebooks, and trained model checkpoints developed as part of this project.



Please refer to the original dataset source for the dataset's applicable terms and licensing conditions.

