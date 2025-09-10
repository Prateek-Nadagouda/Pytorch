# evaluate.py
# Small wrappers around scikit-learn metrics for clarity in training logs.

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report
)

# Compute regression metrics: MSE, MAE, and RMSE.
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mse ** 0.5
    return {"mse": mse, "mae": mae, "rmse": rmse}

# Binary classification metrics: accuracy, precision, recall, f1, optional ROC AUC.
def binary_classification_metrics(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            roc_auc = None
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }

# Multiclass metrics: accuracy + macro/micro averaged precision/recall/f1 and confusion matrix.
def multiclass_classification_metrics(y_true, y_pred, average='macro'):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report_text = classification_report(y_true, y_pred, zero_division=0)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "report_text": report_text
    }
