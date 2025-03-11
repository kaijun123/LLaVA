from sklearn import metrics
import torch


def get_classification(y_pred):
    # Ensure y_pred is a list of tensors
    if isinstance(y_pred, list):
        # Concatenate all tensors along the first dimension
        y_pred_tensor = torch.cat(y_pred, dim=0)
    else:
        y_pred_tensor = y_pred  # In case it's already a tensor

    # Apply sigmoid activation
    sigmoid_data = torch.sigmoid(y_pred_tensor)

    # Round to get binary classification
    classification = torch.round(sigmoid_data)

    # Flatten the tensor
    return classification.flatten()


def flatten(data):
  data = torch.cat(data, dim=0)
  return data.flatten()

def calculate_accuracy(y_pred, y_true):
  classification = get_classification(y_pred)
  y_true = flatten(y_true)
  accuracy_score = metrics.accuracy_score(y_pred=classification, y_true=y_true)
  return accuracy_score

def calculate_f1(y_pred, y_true):
  classification = get_classification(y_pred)
  y_true = flatten(y_true)
  f1_score = metrics.f1_score(y_pred=classification, y_true=y_true)
  return f1_score

def calculate_recall(y_pred, y_true):
  classification = get_classification(y_pred)
  y_true = flatten(y_true)
  recall_score = metrics.recall_score(y_pred=classification, y_true=y_true)
  return recall_score

def calculate_precision(y_pred, y_true):
  classification = get_classification(y_pred)
  y_true = flatten(y_true)
  precision_score = metrics.precision_score(y_pred=classification, y_true=y_true)
  return precision_score