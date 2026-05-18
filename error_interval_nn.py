"""
Νευρωνικό Δίκτυο Πρόβλεψης Χρονικού Διαστήματος T μεταξύ Διαδοχικών Σφαλμάτων
=================================================================================
Αρχιτεκτονική: Sliding-window Feed-Forward Neural Network (FFNN)
- Είσοδος : N προηγούμενες τιμές T  (N = υπερπαράμετρος)
- Έξοδος  : Επόμενη τιμή T (one-step-ahead regression)

Δεδομένα:
  - CSV 1 : Επεισόδια με λάθος ημερομηνίες  → χρησιμοποιείται η ημερ/νία για υπολογισμό T
  - CSV 2 : Επεισόδια με λάθος τιμές κυβικών → χρησιμοποιείται η ημερ/νία για υπολογισμό T

Χρήση:
  python error_interval_nn.py --date_csv dates_errors.csv --value_csv values_errors.csv
  python error_interval_nn.py --date_csv dates_errors.csv --value_csv values_errors.csv \
      --date_col "Ημερομηνία" --window 5 --hidden 64 32 --epochs 200 --lr 0.001
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# 1.  Φόρτωση & Προεπεξεργασία CSV
# ─────────────────────────────────────────────

def load_and_parse(csv_path: str, date_col: str) -> pd.Series:
    """
    Φορτώνει CSV, εντοπίζει τη στήλη ημερομηνίας, επιστρέφει
    ταξινομημένη σειρά ημερομηνιών σφαλμάτων.
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = df.columns.str.strip()

    # Αυτόματος εντοπισμός στήλης ημερομηνίας αν δεν δοθεί
    if date_col not in df.columns:
        candidates = [c for c in df.columns
                      if any(k in c.lower() for k in ["date","ημερ","time","ώρα","datetime"])]
        if not candidates:
            raise ValueError(
                f"Δεν βρέθηκε στήλη ημερομηνίας στο {csv_path}.\n"
                f"Διαθέσιμες στήλες: {list(df.columns)}\n"
                f"Χρησιμοποιήστε --date_col για να ορίσετε τη σωστή στήλη."
            )
        date_col = candidates[0]
        print(f"  ℹ️  Αυτόματη επιλογή στήλης ημερομηνίας: '{date_col}'")

    dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    n_bad = dates.isna().sum()
    if n_bad:
        print(f"  ⚠️  {n_bad} μη αναλύσιμες ημερομηνίες αφαιρέθηκαν.")
    dates = dates.dropna().sort_values().reset_index(drop=True)
    return dates


def compute_intervals(dates: pd.Series) -> np.ndarray:
    """Υπολογίζει T_i = t_{i+1} - t_i σε ώρες (float)."""
    diffs = dates.diff().dropna()
    T = diffs.dt.total_seconds() / 3600.0   # σε ώρες
    T = T[T > 0].values                      # αφαίρεση αρνητικών/μηδενικών (δεδομένα-σφάλματα)
    return T


def build_sequences(T: np.ndarray, N: int):
    """
    Sliding window: κάθε δείγμα είναι (T[i:i+N], T[i+N]).
    Επιστρέφει X (n_samples, N) και y (n_samples,).
    """
    X, y = [], []
    for i in range(len(T) - N):
        X.append(T[i:i+N])
        y.append(T[i+N])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────
# 2.  PyTorch Dataset & Μοντέλο
# ─────────────────────────────────────────────

class IntervalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class IntervalFFNN(nn.Module):
    """
    Feed-Forward Neural Network για πρόβλεψη T.
    Υπερπαράμετροι: N (window), hidden_sizes, dropout.
    """
    def __init__(self, input_size: int, hidden_sizes: list[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))   # έξοδος: scalar T
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 3.  Εκπαίδευση
# ─────────────────────────────────────────────

def train_model(model, train_loader, val_loader,
                epochs: int, lr: float, device: torch.device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15,
                                                            factor=0.5)
    history = {"train": [], "val": []}
    best_val, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        # — Training —
        model.train()
        tr_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(X_b)
        tr_loss /= len(train_loader.dataset)

        # — Validation —
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b)
                vl_loss += criterion(pred, y_b).item() * len(X_b)
        vl_loss /= len(val_loader.dataset)

        history["train"].append(tr_loss)
        history["val"].append(vl_loss)
        scheduler.step(vl_loss)

        if vl_loss < best_val:
            best_val = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"Train MSE={tr_loss:.4f}  Val MSE={vl_loss:.4f}")

    model.load_state_dict(best_state)
    return history


# ─────────────────────────────────────────────
# 4.  Αξιολόγηση & Οπτικοποίηση
# ─────────────────────────────────────────────

def evaluate(model, loader, scaler, device):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            p = model(X_b.to(device)).cpu().numpy()
            preds.append(p)
            actuals.append(y_b.numpy())
    preds   = scaler.inverse_transform(np.vstack(preds))
    actuals = scaler.inverse_transform(np.vstack(actuals))
    mae  = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2   = r2_score(actuals, preds)
    return actuals.flatten(), preds.flatten(), mae, rmse, r2


def plot_results(history, actuals_tr, preds_tr,
                 actuals_te, preds_te, T_raw,
                 source_label: str, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Νευρωνικό Δίκτυο Πρόβλεψης Διαστήματος T — {source_label}",
                 fontsize=13, fontweight="bold")

    # (A) Learning curves
    ax = axes[0, 0]
    ax.plot(history["train"], label="Train MSE")
    ax.plot(history["val"],   label="Val MSE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (κανονικ.)")
    ax.set_title("Καμπύλες Μάθησης"); ax.legend(); ax.grid(True, alpha=0.3)

    # (B) Raw T series
    ax = axes[0, 1]
    ax.plot(T_raw, marker="o", ms=3, linewidth=0.8, color="steelblue")
    ax.set_xlabel("Επεισόδιο i"); ax.set_ylabel("T (ώρες)")
    ax.set_title("Χρονοσειρά T μεταξύ Σφαλμάτων"); ax.grid(True, alpha=0.3)

    # (C) Test: actual vs predicted
    ax = axes[1, 0]
    ax.plot(actuals_te, label="Πραγματικό", color="steelblue")
    ax.plot(preds_te,   label="Πρόβλεψη",  color="tomato", linestyle="--")
    ax.set_xlabel("Δείγμα"); ax.set_ylabel("T (ώρες)")
    ax.set_title("Test Set: Πραγματικό vs Πρόβλεψη"); ax.legend(); ax.grid(True, alpha=0.3)

    # (D) Scatter
    ax = axes[1, 1]
    all_a = np.concatenate([actuals_tr, actuals_te])
    all_p = np.concatenate([preds_tr,   preds_te])
    ax.scatter(actuals_tr, preds_tr, alpha=0.5, s=15, label="Train", color="steelblue")
    ax.scatter(actuals_te, preds_te, alpha=0.7, s=15, label="Test",  color="tomato")
    lims = [min(all_a.min(), all_p.min()), max(all_a.max(), all_p.max())]
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.set_xlabel("Πραγματικό T (ώρες)"); ax.set_ylabel("Προβλεπόμενο T (ώρες)")
    ax.set_title("Scatter: Πραγματικό vs Πρόβλεψη"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"results_{source_label.replace(' ','_')}.png"
    plt.savefig(out_path, dpi=150)
    print(f"  📊 Γράφημα αποθηκεύτηκε: {out_path}")
    plt.close()


# ─────────────────────────────────────────────
# 5.  Κύρια Ροή
# ─────────────────────────────────────────────

def run_pipeline(T_raw: np.ndarray, label: str, args, out_dir: Path):
    print(f"\n{'='*60}")
    print(f"  Δεδομένα: {label}  |  Συνολικά διαστήματα: {len(T_raw)}")
    print(f"  N (window) = {args.window}  |  Κρυφά επίπεδα = {args.hidden}")
    print(f"{'='*60}")

    if len(T_raw) < args.window + 5:
        print(f"  ⚠️  Ανεπαρκή δεδομένα (χρειάζονται > {args.window+5}). Παράλειψη.")
        return

    # Κανονικοποίηση
    scaler = MinMaxScaler()
    T_scaled = scaler.fit_transform(T_raw.reshape(-1, 1)).flatten()

    # Sliding window sequences
    X, y = build_sequences(T_scaled, args.window)
    print(f"  Δείγματα: {len(X)}  (train/val/test ~ 70/15/15)")

    # Διαχωρισμός train / val / test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30,
                                                  shuffle=False)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                                  shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Συσκευή: {device}")

    train_ds = IntervalDataset(X_tr, y_tr)
    val_ds   = IntervalDataset(X_val, y_val)
    test_ds  = IntervalDataset(X_te, y_te)

    train_ldr = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_ldr   = DataLoader(val_ds,   batch_size=args.batch)
    test_ldr  = DataLoader(test_ds,  batch_size=args.batch)

    # Μοντέλο
    model = IntervalFFNN(input_size=args.window,
                         hidden_sizes=args.hidden,
                         dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Παράμετροι μοντέλου: {n_params:,}")
    print(f"\n  Εκπαίδευση...")

    history = train_model(model, train_ldr, val_ldr,
                          args.epochs, args.lr, device)

    # Αξιολόγηση
    a_tr, p_tr, mae_tr, rmse_tr, r2_tr = evaluate(model, train_ldr, scaler, device)
    a_te, p_te, mae_te, rmse_te, r2_te = evaluate(model, test_ldr,  scaler, device)

    print(f"\n  ─── Αποτελέσματα ───")
    print(f"  Train → MAE={mae_tr:.2f}h  RMSE={rmse_tr:.2f}h  R²={r2_tr:.4f}")
    print(f"  Test  → MAE={mae_te:.2f}h  RMSE={rmse_te:.2f}h  R²={r2_te:.4f}")

    # Αποθήκευση μοντέλου
    model_path = out_dir / f"model_{label.replace(' ','_')}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "scaler": scaler,
        "window": args.window,
        "hidden": args.hidden,
        "label":  label,
    }, model_path)
    print(f"  💾 Μοντέλο αποθηκεύτηκε: {model_path}")

    plot_results(history, a_tr, p_tr, a_te, p_te, T_raw, label, out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="FFNN πρόβλεψης χρονικού διαστήματος T μεταξύ σφαλμάτων"
    )
    parser.add_argument("--date_csv",   required=True,
                        help="CSV με λάθος ημερομηνίες (για εξαγωγή T)")
    parser.add_argument("--value_csv",  required=True,
                        help="CSV με λάθος τιμές κυβικών (για εξαγωγή T)")
    parser.add_argument("--date_col",   default="",
                        help="Όνομα στήλης ημερομηνίας (κοινό και για τα 2 CSVs)")
    parser.add_argument("--window",     type=int,   default=5,
                        help="N: μέγεθος παραθύρου (υπερπαράμετρος, default=5)")
    parser.add_argument("--hidden",     type=int, nargs="+", default=[64, 32],
                        help="Νευρώνες κρυφών επιπέδων (default: 64 32)")
    parser.add_argument("--dropout",    type=float, default=0.2,
                        help="Dropout rate (default=0.2)")
    parser.add_argument("--epochs",     type=int,   default=200,
                        help="Εποχές εκπαίδευσης (default=200)")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Learning rate (default=0.001)")
    parser.add_argument("--batch",      type=int,   default=16,
                        help="Batch size (default=16)")
    parser.add_argument("--out_dir",    default="results",
                        help="Φάκελος αποθήκευσης αποτελεσμάτων")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n🔍 Φόρτωση δεδομένων...")

    date_col = args.date_col if args.date_col else ""

    # CSV 1: Λάθος ημερομηνίες
    print(f"\n📂 CSV λάθος ημερομηνιών: {args.date_csv}")
    dates1 = load_and_parse(args.date_csv, date_col)
    T1 = compute_intervals(dates1)
    print(f"  → {len(dates1)} εγγραφές, {len(T1)} διαστήματα T  "
          f"[{T1.min():.1f}h – {T1.max():.1f}h, μέση={T1.mean():.1f}h]")
    run_pipeline(T1, "Λάθος Ημερομηνίες", args, out_dir)

    # CSV 2: Λάθος τιμές κυβικών
    print(f"\n📂 CSV λάθος τιμών κυβικών: {args.value_csv}")
    dates2 = load_and_parse(args.value_csv, date_col)
    T2 = compute_intervals(dates2)
    print(f"  → {len(dates2)} εγγραφές, {len(T2)} διαστήματα T  "
          f"[{T2.min():.1f}h – {T2.max():.1f}h, μέση={T2.mean():.1f}h]")
    run_pipeline(T2, "Λάθος Τιμές Κυβικών", args, out_dir)

    # Προαιρετικά: συνδυασμένο pipeline αν θέλεις ενιαίο μοντέλο
    T_all = np.concatenate([T1, T2])
    T_all_sorted = np.sort(T_all)
    print(f"\n📊 Συνδυασμένα διαστήματα: {len(T_all_sorted)}")
    run_pipeline(T_all_sorted, "Συνδυασμένα Σφάλματα", args, out_dir)

    print(f"\n✅ Ολοκλήρωση. Αποτελέσματα στον φάκελο: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
