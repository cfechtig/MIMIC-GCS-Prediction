
import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from models.hybrid_model import HybridModel
from train_eval.train import train_model, get_loss_fn
from train_eval.eval import evaluate_model
from utils.plotting import plot_loss_curves, plot_roc_curve, plot_confusion_matrix

def balance_training_set(df_train, target, task):
    if task == "classification":
        # Binary case
        minority = df_train[df_train[target] == 1]
        majority = df_train[df_train[target] == 0].sample(n=len(minority), random_state=42)
        df_train = pd.concat([minority, majority]).sample(frac=1, random_state=42)
    elif task == "multiclass":
        # Multiclass case: balance across all classes
        counts = df_train[target].value_counts()
        min_count = counts.min()
        df_train = pd.concat([
            df_train[df_train[target] == cls].sample(min_count, random_state=42)
            for cls in counts.index
        ]).sample(frac=1, random_state=42)
    return df_train


def run_experiment(df, static_features, sequential_features, target, config, balance_data=False):
    os.makedirs(config["save_dir"], exist_ok=True)
    task = config["task"]

    # === Split data ===
    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]
    df_test = df[df["split"] == "test"]

    if balance_data and task in ["classification", "multiclass"]:
        df_train = balance_training_set(df_train, target, task)

    # === Scale static & sequential features ===
    scaler_static = StandardScaler()
    scaler_seq = StandardScaler()
    scaler_static.fit(df_train[static_features])
    scaler_seq.fit(df_train[sequential_features])

    def prepare_inputs(df_sub):
        X_static = scaler_static.transform(df_sub[static_features])
        X_seq = scaler_seq.transform(df_sub[sequential_features])
        time_steps = sorted(set(int(col.split('_t')[1]) for col in sequential_features))
        features = sorted(set(col.split('_t')[0] for col in sequential_features))
        X_seq = X_seq.reshape((X_seq.shape[0], len(time_steps), len(features)))
        y = df_sub[target].values
        if task == "classification":
            y = (y >= 1).astype(int)
        elif task == "multiclass":
            y = y.astype(int)
        return torch.tensor(X_static).float(), torch.tensor(X_seq).float(), torch.tensor(y)

    X_static_train, X_seq_train, y_train = prepare_inputs(df_train)
    X_static_val, X_seq_val, y_val = prepare_inputs(df_val)
    X_static_test, X_seq_test, y_test = prepare_inputs(df_test)

    print("Train labels:", np.unique(y_train.numpy(), return_counts=True))
    print("Val labels:", np.unique(y_val.numpy(), return_counts=True))
    print("Test labels:", np.unique(y_test.numpy(), return_counts=True))
    print()

    # === Build Dataloaders ===
    def make_loader(X_static, X_seq, y):
        y_tensor = y.float() if task != "multiclass" else y.long()
        return DataLoader(TensorDataset(X_static, X_seq, y_tensor),
                          batch_size=config["batch_size"], shuffle=(y is y_train))

    train_dl = make_loader(X_static_train, X_seq_train, y_train)
    val_dl = make_loader(X_static_val, X_seq_val, y_val)
    test_dl = make_loader(X_static_test, X_seq_test, y_test)

    # === Model ===
    model = HybridModel(
        static_dim=X_static_train.shape[1],
        seq_feature_dim=X_seq_train.shape[2],
        task=task,
        lstm_hidden=config["lstm_hidden"],
        static_hidden=config["static_hidden"],
        dropout=config["dropout"],
        num_classes=config.get("num_classes", 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = get_loss_fn(task=task)

    # === Train ===
    model, history = train_model(
        model, train_dl, val_dl, optimizer, loss_fn,
        task=task,
        n_epochs=config["epochs"],
        save_path=os.path.join(config["save_dir"], "best_model.pt")
    )

    # === Evaluate ===
    best_model_path = os.path.join(config["save_dir"], "best_model.pt")
    model.load_state_dict(torch.load(best_model_path))
    results, y_true, y_pred = evaluate_model(model, test_dl, task=task)

    # === Plot ===
    plot_loss_curves(history, save_path=os.path.join(config["save_dir"], "loss_curve.png"))
    if task == "classification":
        plot_roc_curve(y_true, y_pred, save_path=os.path.join(config["save_dir"], "roc.png"))
        plot_confusion_matrix(y_true, y_pred >= 0.5, labels=["No Drop", "Drop"],
                              save_path=os.path.join(config["save_dir"], "confusion_matrix.png"))
    elif task == "multiclass":
        y_pred_classes = y_pred.argmax(axis=1)
        plot_confusion_matrix(y_true, y_pred_classes,
                              labels=[str(i) for i in range(config.get("num_classes", 4))],
                              save_path=os.path.join(config["save_dir"], "confusion_matrix.png"))

    # === Save results ===
    with open(os.path.join(config["save_dir"], "metrics.json"), "w") as f:
        json.dump({k: str(v) for k, v in results.items()}, f, indent=2)
    with open(os.path.join(config["save_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return results
