import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

#set the seed for reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_patient_tables(data_dir: Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    metadata = pd.read_csv(data_dir / "metadata.csv")
    vitals = pd.read_csv(data_dir / "vitals_timeseries.csv")
    notes = pd.read_csv(data_dir / "clinical_notes.csv")

    metadata = metadata.sort_values("Patient_ID").reset_index(drop=True) # Ensure patients are in order.
    patient_ids = metadata["Patient_ID"].tolist()
    labels = metadata["Label"].astype(np.float32).to_numpy() # Extract the 'target' (0 or 1).

    vitals_features = ["HR", "RR", "SpO2", "SBP"]
    vitals_arr = np.zeros((len(patient_ids), 24, len(vitals_features)), dtype=np.float32)

    vitals_sorted = vitals.sort_values(["Patient_ID", "Hour"]) # Sort vitals chronologically.
    for pid, group in vitals_sorted.groupby("Patient_ID"):
        block = group[vitals_features].ffill().bfill().fillna(0.0) # Fill missing values.
        vitals_arr[int(pid)] = block.to_numpy(dtype=np.float32)

    notes_sorted = notes.sort_values(["Patient_ID", "Hour"]) # Sort notes chronologically.
    notes_per_patient: Dict[int, str] = {}
    for pid, group in notes_sorted.groupby("Patient_ID"):
        # Join all notes for one patient into one long string, separated by [SEP] (the BERT separator).
        text = " [SEP] ".join(group["Note_Content"].astype(str).tolist())
        notes_per_patient[int(pid)] = text
    # Create the final list of texts matching the patient IDs.
    note_texts = [notes_per_patient.get(int(pid), "") for pid in patient_ids]
    return vitals_arr, note_texts, labels


class ClinicalFusionDataset(Dataset):
    def __init__(self, vitals: np.ndarray, notes: List[str], labels: np.ndarray):
        self.vitals = torch.tensor(vitals, dtype=torch.float32)
        self.notes = notes
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # Return data for one specific patient when asked.
        return self.vitals[idx], self.notes[idx], self.labels[idx]


class NotesOnlyModel(nn.Module):
    def __init__(
        self,
        bert_model_name: str,
        dropout: float = 0.2,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(bert_model_name)
        bert_hidden = self.text_encoder.config.hidden_size

        if freeze_bert:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(bert_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = text_outputs.last_hidden_state[:, 0, :]
        logits = self.head(cls_embedding).squeeze(-1)
        return logits


class VitalsOnlyModel(nn.Module):
    def __init__(
        self,
        gru_hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vitals_encoder = nn.GRU(
            input_size=4,  # HR, RR, SpO2, SBP
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(gru_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, vitals):
        _, h_n = self.vitals_encoder(vitals)
        vitals_embedding = h_n[-1]
        logits = self.head(vitals_embedding).squeeze(-1)
        return logits


class MultimodalFusionModel(nn.Module):
    def __init__(
        self,
        bert_model_name: str,
        gru_hidden_dim: int = 64,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.2,
        freeze_bert: bool = False,
    ):
        super().__init__()
        # Tower 1: Text Encoder (ClinicalBERT)
        self.text_encoder = AutoModel.from_pretrained(bert_model_name)
        bert_hidden = self.text_encoder.config.hidden_size

        if freeze_bert: # If true, we don't update BERT weights (saves time/memory).
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Tower 2: Vitals Encoder (GRU)
        self.vitals_encoder = nn.GRU(
            input_size=4, # 4 vital signs (HR, RR, SpO2, SBP).
            hidden_size=gru_hidden_dim, # Number of features in the hidden state, size of the memory.
            num_layers=1, # Simple single-layer GRU.
            batch_first=True,# Input format is (Batch, Time, Features).
            bidirectional=False,    # Process time forward only (like a real patient stay).
        )

        # The Fusion Head: Combines both towers.
        
        self.fusion = nn.Sequential(
            nn.Linear(bert_hidden + gru_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1), #outputs the logits
        )

    def forward(self, input_ids, attention_mask, vitals):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # We take the [CLS] token (index 0), which BERT uses to represent the whole sentence.
        cls_embedding = text_outputs.last_hidden_state[:, 0, :]

        _, h_n = self.vitals_encoder(vitals)
        vitals_embedding = h_n[-1] # Take the 'last' state of the memory 

        fused = torch.cat([cls_embedding, vitals_embedding], dim=1)
        # Calculate the final score.
        logits = self.fusion(fused).squeeze(-1)
        return logits


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch):
        vitals, notes, labels = zip(*batch)
        vitals = torch.stack(vitals)
        labels = torch.stack(labels)
        tokenized = tokenizer(
            list(notes),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return tokenized["input_ids"], tokenized["attention_mask"], vitals, labels

    return collate


def evaluate_multimodal(model, dataloader, device, loss_fn):
    model.eval()
    losses, probs_all, labels_all = [], [], []
    with torch.no_grad():
        for input_ids, attention_mask, vitals, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            vitals = vitals.to(device)
            labels = labels.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, vitals=vitals)
            loss = loss_fn(logits, labels)
            # Turn into a probability percentage.
            probs = torch.sigmoid(logits)

            losses.append(loss.item())
            probs_all.extend(probs.detach().cpu().numpy().tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())

    auc = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else float("nan")
    return float(np.mean(losses)), float(auc)


def evaluate_notes_only(model, dataloader, device, loss_fn):
    model.eval()
    losses, probs_all, labels_all = [], [], []
    with torch.no_grad():
        for input_ids, attention_mask, _, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            probs = torch.sigmoid(logits)

            losses.append(loss.item())
            probs_all.extend(probs.detach().cpu().numpy().tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())

    auc = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else float("nan")
    return float(np.mean(losses)), float(auc)


def evaluate_vitals_only(model, dataloader, device, loss_fn):
    model.eval()
    losses, probs_all, labels_all = [], [], []
    with torch.no_grad():
        for _, _, vitals, labels in dataloader:
            vitals = vitals.to(device)
            labels = labels.to(device)

            logits = model(vitals=vitals)
            loss = loss_fn(logits, labels)
            probs = torch.sigmoid(logits)

            losses.append(loss.item())
            probs_all.extend(probs.detach().cpu().numpy().tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())

    auc = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else float("nan")
    return float(np.mean(losses)), float(auc)


def train_one_mode(args, mode: str) -> float:
    """
    mode: "multimodal" | "notes_only" | "vitals_only"
    returns: best validation AUC
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    # SETUP DATA
    vitals, notes, labels = build_patient_tables(data_dir)
    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
    train_ds = ClinicalFusionDataset(vitals[train_idx], [notes[i] for i in train_idx], labels[train_idx])
    val_ds = ClinicalFusionDataset(vitals[val_idx], [notes[i] for i in val_idx], labels[val_idx])

    collate_fn = make_collate_fn(tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # SETUP MODEL
    if mode == "multimodal":

        model = MultimodalFusionModel(
            bert_model_name=args.bert_model_name,
            gru_hidden_dim=args.gru_hidden_dim,
            fusion_hidden_dim=args.fusion_hidden_dim,
            dropout=args.dropout,
            freeze_bert=args.freeze_bert,
        ).to(device)
        evaluate_fn = evaluate_multimodal
        train_step_requires = "both"
    elif mode == "notes_only":

        model = NotesOnlyModel(
            bert_model_name=args.bert_model_name,
            dropout=args.dropout,
            freeze_bert=args.freeze_bert,
        ).to(device)
        evaluate_fn = evaluate_notes_only
        train_step_requires = "notes"
    elif mode == "vitals_only":

        model = VitalsOnlyModel(
            gru_hidden_dim=args.gru_hidden_dim,
            dropout=args.dropout,
        ).to(device)
        evaluate_fn = evaluate_vitals_only
        train_step_requires = "vitals"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Handle class imbalance (same for all modes).
    train_pos = labels[train_idx].sum()
    train_neg = len(train_idx) - train_pos
    pos_weight = torch.tensor([(train_neg / max(train_pos, 1.0))], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / f"best_{mode}_model.pt"


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []

        for batch in train_loader:
            optimizer.zero_grad()

            if train_step_requires == "both":
                input_ids, attention_mask, vitals_batch, labels_batch = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                vitals_batch = vitals_batch.to(device)
                labels_batch = labels_batch.to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask, vitals=vitals_batch)
            elif train_step_requires == "notes":
                input_ids, attention_mask, _, labels_batch = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels_batch = labels_batch.to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            elif train_step_requires == "vitals":
                _, _, vitals_batch, labels_batch = batch
                vitals_batch = vitals_batch.to(device)
                labels_batch = labels_batch.to(device)
                logits = model(vitals=vitals_batch)
            else:
                raise RuntimeError("Invalid train_step_requires")

            loss = loss_fn(logits, labels_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        val_loss, val_auc = evaluate_fn(model, val_loader, device, loss_fn)
        print(
            f"[{mode}] Epoch {epoch:02d} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_auc={val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "val_auc": best_auc,
                    "mode": mode,
                },
                best_ckpt,
            )

    print(f"[{mode}] Best validation AUC: {best_auc:.4f}")
    print(f"[{mode}] Saved best checkpoint: {best_ckpt}")
    return float(best_auc)


def predict_patient_probability(
    model: nn.Module,
    tokenizer,
    note_text: str,
    vitals_24x4: np.ndarray,
    device: torch.device,
    max_length: int = 256,
    mode: str = "multimodal",
) -> float:
    model.eval()
    with torch.no_grad():
        tok = tokenizer(
            note_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        vitals = torch.tensor(vitals_24x4, dtype=torch.float32).unsqueeze(0)
        if mode == "multimodal":
            logits = model(
                input_ids=tok["input_ids"].to(device),
                attention_mask=tok["attention_mask"].to(device),
                vitals=vitals.to(device),
            )
        elif mode == "notes_only":
            logits = model(
                input_ids=tok["input_ids"].to(device),
                attention_mask=tok["attention_mask"].to(device),
            )
        elif mode == "vitals_only":
            logits = model(vitals=vitals.to(device))
        else:
            raise ValueError(f"Unknown mode: {mode}")
        # Turn into a probability percentage.
        prob = torch.sigmoid(logits).item()
    return float(prob)


def parse_args():
    parser = argparse.ArgumentParser(description="ClinicalBERT + GRU multimodal fusion for deterioration prediction.")
    parser.add_argument("--data_dir", type=str, default="patient_data", help="Path containing metadata/vitals/notes CSVs.")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory for saved model checkpoint.")
    parser.add_argument("--bert_model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gru_hidden_dim", type=int, default=64)
    parser.add_argument("--fusion_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze_bert",
        action="store_true",
        help="Freeze ClinicalBERT weights and train only GRU/fusion layers.",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="multimodal",
        choices=["multimodal", "vitals_only", "notes_only", "all"],
        help="Run a single modality setting or train all three and compare.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.ablation == "all":
        modes = ["notes_only", "vitals_only", "multimodal"]
        results = {}
        for m in modes:
            print("\n" + "="*50)
            print(f" EXPERIMENT: {m.upper()} ")
            print("="*50 + "\n")
            results[m] = train_one_mode(args, m)

        print("\nAblation comparison (best validation ROC-AUC):")
        for m in modes:
            print(f"- {m}: {results[m]:.4f}")

        best_mode = max(results, key=results.get)
        print(f"\nBest model: {best_mode} (AUC={results[best_mode]:.4f})")
    else:
        train_one_mode(args, args.ablation)
