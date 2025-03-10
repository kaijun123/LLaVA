from sklearn import metrics
import torch

def get_classification(y_pred):
  y_pred = torch.Tensor(y_pred)
  sigmoid_data = torch.sigmoid(y_pred)
  classification = torch.round(sigmoid_data)
  # print("classification:", classification)
  return torch.flatten(classification, start_dim=0)

def flatten(data):
  data = torch.Tensor(data)
  return torch.flatten(data, start_dim=0)

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