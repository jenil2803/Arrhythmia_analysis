# Arrhythmia Analysis Using 12-Lead ECG Signals

This project builds a deep learning pipeline for multi-class arrhythmia classification from raw 12-lead ECG records. It is implemented in a Jupyter notebook and is designed to run in Google Colab with data stored in Google Drive.

The notebook performs:
- Dataset setup and extraction from a ZIP archive
- Header-based diagnosis parsing from WFDB-style `.hea` files
- Signal loading and normalization from `.mat` files
- Label filtering to five target cardiac rhythm classes
- Model training using a hybrid deep architecture (SE-ResNet + BiLSTM + Attention)
- Model evaluation with classification report, confusion matrix, and ROC curves

## Project Goal

The goal is to classify ECG recordings into five clinically relevant rhythm classes using supervised learning and a robust feature extractor for temporal biomedical signals.

Target SNOMED CT classes used in the notebook:
- `426177001` -> Sinus Bradycardia (SB)
- `426783006` -> Sinus Rhythm (SR)
- `164890007` -> Atrial Flutter (AFL)
- `427084000` -> Sinus Tachycardia (ST)
- `164889003` -> Atrial Fibrillation (AFIB)

## Workflow Summary

### 1. Environment and Data Setup

The notebook:
- Mounts Google Drive in Colab
- Looks for a ZIP file at:
  - `/content/drive/MyDrive/AML/WFDBRecords.zip`
- Extracts it to:
  - `/content/ecg_data`
- Recursively verifies that `.mat` and `.hea` files are present

This helps avoid repeated extraction and improves I/O performance compared with reading directly from Drive.

### 2. Data Parsing and Label Construction

Each ECG sample is built from:
- Signal file (`.mat`) containing multi-lead waveform data
- Header file (`.hea`) containing diagnosis codes

Key processing logic:
- Signals are loaded with `scipy.io.loadmat`
- Data shape is corrected to `(12, sequence_length)` when needed
- Per-lead z-score normalization is applied
- Diagnosis is extracted via regex from `#Dx:` entries in headers
- Only records whose primary diagnosis belongs to the five target classes are retained

The pipeline also exports a full diagnosis audit file:
- `patient_diagnosis_status.csv`

This CSV records each patient/file with one of:
- Target class name
- `Other/Ignored`
- `NA` (missing or unparseable diagnosis)

## Model Architecture

The classification model is `Advanced_ECG_Net`, designed for long 1D biomedical sequences.

### Components

1. Initial temporal feature extraction
- `Conv1d(12 -> 64)` with large kernel
- Batch normalization + ReLU
- Max pooling

2. Residual feature hierarchy
- Three 1D ResNet stages
- Residual skip connections for stable deep training
- Channel recalibration through Squeeze-and-Excitation (SE) blocks

3. Sequence modeling
- Bidirectional LSTM over temporal embeddings
- Captures forward and backward context in ECG dynamics

4. Temporal attention
- Learns importance weights across timesteps
- Aggregates sequence into an attention-weighted context vector

5. Classifier head
- Fully connected layers with BatchNorm, ReLU, Dropout
- Final logits for multi-class prediction

## Training Strategy

Configured defaults:
- Batch size: `64`
- Epochs: `35`
- Learning rate: `3e-4`
- Train/Test split: `70/30`

Optimization and regularization:
- Optimizer: Adam
- Loss: weighted cross entropy (class imbalance handling)
- Scheduler: `ReduceLROnPlateau` to reduce LR when validation loss stalls

Class weights are computed from encoded labels using balanced weighting.

## Evaluation Outputs

After training, the notebook generates:
- Per-class precision, recall, f1-score (`classification_report`)
- Confusion matrix heatmap
- One-vs-rest ROC curve for each class with AUC values

These outputs provide both threshold-based and ranking-based views of model quality.

## Repository Structure

Current workspace:
- `FINAL.ipynb` - End-to-end data setup, training, and evaluation notebook
- `Arrhythmia_Analysis.pdf` - Project report/documentation artifact
- `README.md` - Project overview and usage guide

## How To Run

### Option A: Google Colab (recommended)

1. Upload the dataset ZIP to Google Drive:
- `/content/drive/MyDrive/AML/WFDBRecords.zip`

2. Open `FINAL.ipynb` in Colab.

3. Run cells in order:
- Mount and extract data
- Initialize dataset and model classes
- Start training via `run_training()`

4. Review generated outputs and charts.

### Option B: Local Jupyter (with path edits)

If running locally, update path config variables in the notebook:
- `ZIP_PATH`
- `EXTRACT_PATH`
- `CONFIG["data_path"]`
- `CONFIG["conditions_file"]`

## Dependencies

Main Python libraries used:
- `torch`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`

Colab usually includes most dependencies by default. Install missing ones with `pip` if needed.

## Notes and Assumptions

- The parser currently uses the first diagnosis code as the primary label.
- Samples with diagnosis codes outside the five target classes are excluded from training.
- Corrupt/missing signal files are replaced with zero tensors as a fallback.

## Potential Improvements

- Add patient-level split to prevent leakage across train/test sets
- Add deterministic seed control for full reproducibility
- Save model checkpoints and best epoch selection
- Add early stopping and richer experiment tracking
- Extend to multi-label classification for multi-diagnosis headers
- Add inference script for single-file prediction


## Author

Prepared for ECG arrhythmia classification experiments using WFDB-style records and deep learning in PyTorch.
