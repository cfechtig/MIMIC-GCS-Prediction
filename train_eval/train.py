import torch
import torch.nn as nn
from collections import defaultdict

def get_loss_fn(task):
    if task == "regression":
        return nn.MSELoss()
    elif task == "classification":
        return nn.BCELoss()
    elif task == "multiclass":
        return nn.NLLLoss()
    else:
        raise ValueError(f"Unknown task type: {task}")

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    task="regression",
    n_epochs=20,
    device=None,
    save_path=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = defaultdict(list)
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(1, n_epochs + 1):
        # Training
        model.train()
        train_loss = 0
        for xb_static, xb_seq, yb in train_loader:
            xb_static, xb_seq, yb = xb_static.to(device), xb_seq.to(device), yb.to(device).long() if task == "multiclass" else yb.to(device)
            pred = model(xb_static, xb_seq)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb_static.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb_static, xb_seq, yb in val_loader:
                xb_static, xb_seq, yb = xb_static.to(device), xb_seq.to(device), yb.to(device).long() if task == "multiclass" else yb.to(device)
                pred = model(xb_static, xb_seq)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb_static.size(0)
        val_loss /= len(val_loader.dataset)

        # Tracking
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    if save_path:
        torch.save(best_model_state, save_path)

    return model, history
