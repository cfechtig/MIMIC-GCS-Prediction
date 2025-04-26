import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

def evaluate_model(model, dataloader, task="regression", threshold=0.5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb_static, xb_seq, yb in dataloader:
            xb_static, xb_seq = xb_static.to(device), xb_seq.to(device)
            preds = model(xb_static, xb_seq).cpu()
            all_preds.append(preds)
            all_labels.append(yb)

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    results = {}

    if task == "regression":
        results["MAE"] = mean_absolute_error(y_true, y_pred)
        results["MSE"] = mean_squared_error(y_true, y_pred)
        results["R2"] = r2_score(y_true, y_pred)

    elif task == "classification":
        y_pred_bin = (y_pred >= threshold).astype(int)
        results["Accuracy"] = accuracy_score(y_true, y_pred_bin)
        results["Precision"] = precision_score(y_true, y_pred_bin)
        results["Recall"] = recall_score(y_true, y_pred_bin)
        results["F1"] = f1_score(y_true, y_pred_bin)
        results["ROC_AUC"] = roc_auc_score(y_true, y_pred)
        results["ConfusionMatrix"] = confusion_matrix(y_true, y_pred_bin)

    elif task == "multiclass":
        y_pred_classes = y_pred.argmax(axis=1)
        results["Accuracy"] = accuracy_score(y_true, y_pred_classes)
        results["Precision_macro"] = precision_score(y_true, y_pred_classes, average="macro")
        results["Recall_macro"] = recall_score(y_true, y_pred_classes, average="macro")
        results["F1_macro"] = f1_score(y_true, y_pred_classes, average="macro")
        results["ConfusionMatrix"] = confusion_matrix(y_true, y_pred_classes)

    return results, y_true, y_pred
