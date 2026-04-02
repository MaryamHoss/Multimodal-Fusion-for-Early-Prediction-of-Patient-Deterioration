# Multimodal Fusion Dataset Generator

This project contains:

- `generate_dataset.py`: generates a synthetic multimodal clinical dataset into `patient_data/`
- `train_fusion_model.py`: trains PyTorch models (ClinicalBERT / GRU / Fusion) to predict patient deterioration probability

The script creates a multimodal clinical dataset for `1000` patients across a `24`-hour observation window with:

- **Static structured features**: age, gender, history of heart failure
- **Time-series vital signs**: heart rate (`HR`), respiratory rate (`RR`), oxygen saturation (`SpO2`), systolic blood pressure (`SBP`)
- **Text notes**: neutral, stable, and deteriorating clinical note trajectories
- **Target label**: `0` (stable) or `1` (deteriorating), with class imbalance (`~10%` deteriorating)

It also simulates real-world data issues by injecting missing values in vitals and then preprocessing with forward fill + backward fill.

## How `generate_dataset.py` works

1. Randomly assigns each patient a stability label using an imbalanced class distribution.
2. Generates static demographics/comorbidity data.
3. Simulates hourly vital sign trajectories:
   - Stable patients have near-baseline noisy signals.
   - Deteriorating patients trend toward physiologic decline (e.g., rising HR/RR, falling SpO2/SBP).
4. Introduces `10%` missing values in time-series data.
5. Creates temporal clinical note sequences with realistic noise and progression.
6. Preprocesses missing vital signs using LOCF-style filling (`ffill().bfill()`).
7. Saves the dataset to `patient_data/` as CSV files.

## Output files (`patient_data/`)

Running `generate_dataset.py` creates a folder `patient_data/` with:

- `metadata.csv`: one row per patient with demographics + label
  - Columns: `Age`, `Gender`, `Hx_Heart_Failure`, `Patient_ID`, `Label`
- `vitals_timeseries.csv`: long-format hourly vitals for each patient
  - Columns: `Patient_ID`, `Hour`, `HR`, `RR`, `SpO2`, `SBP`
- `clinical_notes.csv`: long-format hourly notes for each patient
  - Columns: `Patient_ID`, `Hour`, `Note_Content`

## Environment setup

A local virtual environment is expected at `.venv`.

### Activate the environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies (if needed)

```powershell
python -m pip install numpy pandas
python -m pip install torch transformers scikit-learn
```

## Run the generator

```powershell
python generate_dataset.py
```

After this completes, you should have `patient_data/metadata.csv`, `patient_data/vitals_timeseries.csv`, and `patient_data/clinical_notes.csv`.

## Train the fusion model (PyTorch)

`train_fusion_model.py` trains models to predict the probability of deterioration (`Label=1`) from:

- **Clinical notes** encoded by ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`)
- **Vitals time series** encoded by a GRU
- **Fusion**: concatenates both embeddings and passes them through an MLP head

## Model details (inputs → outputs)

### Inputs

The training script loads one example per **patient** by joining the CSVs in `patient_data/`:

- **Vitals**: a fixed 24-hour window of 4 vital signs:
  - Tensor shape: `(batch, 24, 4)`
  - Feature order: `HR`, `RR`, `SpO2`, `SBP`
  - Missing values are handled with forward/backward filling per patient (`ffill().bfill()`).

- **Clinical notes**: all hourly notes for a patient concatenated into one string:
  - `"note@hour0 [SEP] note@hour1 [SEP] ... [SEP] note@hour23"`
  - Tokenized with ClinicalBERT tokenizer (padding + truncation) to:
    - `input_ids` shape: `(batch, seq_len)`
    - `attention_mask` shape: `(batch, seq_len)`

### Outputs

- The model outputs a **single logit per patient** (binary classification):
  - `logits` shape: `(batch,)`
- The predicted **probability of deterioration** is:
  - `p = sigmoid(logit)` in \([0, 1]\)

### Architecture

`train_fusion_model.py` supports three architectures:

- **Notes-only** (`--ablation notes_only`)
  - ClinicalBERT encoder → `[CLS]` embedding → linear classification head

- **Vitals-only** (`--ablation vitals_only`)
  - GRU encoder over 24×4 vitals → last hidden state → linear classification head

- **Multimodal fusion** (`--ablation multimodal`)
  - ClinicalBERT `[CLS]` embedding (notes) + GRU last hidden state (vitals)
  - Concatenation → MLP fusion head → logit

### Loss + metric

- Training uses `BCEWithLogitsLoss` with `pos_weight` to account for class imbalance.
- Validation metric reported is **ROC-AUC**.

### Multimodal training (notes + vitals)

```powershell
python train_fusion_model.py --data_dir patient_data --ablation multimodal --epochs 3 --batch_size 8
```

Checkpoints are saved to `checkpoints/` (best model by validation ROC-AUC).

## Ablation studies

To compare unimodal vs multimodal performance, use `--ablation`:

- `notes_only`: ClinicalBERT only
- `vitals_only`: GRU only
- `multimodal`: ClinicalBERT + GRU fusion
- `all`: trains all three and prints a comparison

### Run all three and compare

```powershell
python train_fusion_model.py --data_dir patient_data --ablation all --epochs 3 --batch_size 8
```

### Run single modality

```powershell
python train_fusion_model.py --data_dir patient_data --ablation vitals_only --epochs 3
python train_fusion_model.py --data_dir patient_data --ablation notes_only --epochs 3
```

Tip: `--freeze_bert` can speed up experiments by training only the GRU/fusion heads:

```powershell
python train_fusion_model.py --data_dir patient_data --ablation notes_only --freeze_bert --epochs 3
```

## Web UI (Streamlit)

This repo includes a Streamlit UI in `app.py` that:

- Loads the **best multimodal checkpoint** from `checkpoints/best_multimodal_model.pt`
- Lets you select a `Patient_ID` from `patient_data/`
- Visualizes 24-hour vitals
- Lets you edit a note and run inference to display **deterioration probability**

### Install UI dependencies

```powershell
python -m pip install streamlit matplotlib
```

### Run the UI

```powershell
streamlit run app.py
```

If the UI errors due to missing checkpoint, train a multimodal model first (or run ablation `all`) so `checkpoints/best_multimodal_model.pt` exists.

## Notes

- The first run that uses ClinicalBERT will download model weights from Hugging Face and can take a while on Windows.
- If you see a symlink caching warning from `huggingface_hub`, it is safe to ignore (it only affects cache efficiency).

- Number of generated patients
- Shape of cleaned time-series vitals
- Shape of static features
- Class distribution
- Example of the last 3 clinical notes for one patient
