# Plant Disease Classification

A ResNet-50 image classifier with a Streamlit interface. Training uses the official PlantVillage train/test split, stores the class mapping and preprocessing settings in the checkpoint, and saves the best validation checkpoint atomically.

## Requirements

- Windows 11 and Python 3.13
- `uv` for creating the isolated environment
- At least 5 GB of free disk space if you plan to retrain the model
- Internet access is only needed to download training data and, by default, ImageNet pretrained weights

A trained inference checkpoint, `plant_disease_resnet50.pth`, is included in the project folder. The app can make predictions with this model after setup; you do not need to download the dataset or train a model first. The PlantVillage archive is about 2.18 GB and is downloaded from the pinned Hugging Face revision only when training.

## Setup

Run these commands from the project folder in PowerShell. A separate environment avoids replacing an existing `.venv`:

```powershell
uv python install 3.13
uv venv --python 3.13 .venv313
uv pip install --python .\.venv313\Scripts\python.exe -r requirements.txt
```

## Optional: Train or Retrain

The included checkpoint is ready for prediction. To train a new model or replace it, download the archive and train for five epochs:

```powershell
.\.venv313\Scripts\python.exe train.py --download-data --epochs 5
```

The trainer uses the official leaf-level train/test split, reports loss and accuracy each epoch, and writes the best validation checkpoint to `plant_disease_resnet50.pth`. Reruns reuse the downloaded archive and split files. To train without downloading ImageNet weights, pass `--no-pretrained`; training from scratch may require more epochs and produce lower accuracy.

## Run the App

```powershell
.\.venv313\Scripts\python.exe -m streamlit run app.py
```

Open the local URL (http://localhost:8501/) printed by Streamlit. The app loads the included checkpoint automatically. If it is missing or invalid, the app opens its setup screen, where you can train a model or import a trusted checkpoint.

When training is needed, choose "Download data and train" to start a separate local training process. The page shows the current job status and the latest training output; it switches to prediction mode once the first validated checkpoint is saved. Keep the computer on. CPU-only training may take hours or longer. Closing the browser does not stop training.

Training jobs started through the app are protected against duplicate workers. Logs and job status are stored in `.training/`. A failed or interrupted job can be retried from the setup screen; available downloads and extracted data are reused. Restarting training starts a new optimization run, not a resumption of the previous epoch.

Alternatively, import a trusted `.pth` checkpoint produced by this project using "Validate and install". Its metadata and all model weight shapes are checked before it atomically replaces the installed checkpoint. A generic ImageNet ResNet-50 checkpoint is not a plant-disease classifier and is not compatible. A trained checkpoint is sufficient for predictions; the dataset is only required for training.

## Verify

```powershell
.\.venv313\Scripts\python.exe -m unittest discover -s tests -v
```

These tests cover setup and retry states, duplicate-job locking, training status, checkpoint validation and installation, and dataset path safety. They do not establish prediction accuracy; review the validation results from a completed training run before relying on predictions.

## Dataset and limitations

The dataset is published by the Digital Epidemiology Lab at EPFL under CC BY-SA 3.0. This project pins dataset revision `9e97599868962bd0079b8db4b7f1efa9185fa1e7` and uses its published color-image splits. Retain the dataset attribution and license when redistributing derived data.

The current training workflow uses the official test split to select the best checkpoint. Consequently, its displayed validation accuracy and evaluation report are not independent final-test estimates and may be optimistically biased. Use a separate untouched holdout set for an unbiased final evaluation. PlantVillage is a controlled-image research dataset; predictions are not a confirmed diagnosis and may not generalize to field conditions.
