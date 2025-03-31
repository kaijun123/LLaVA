if __name__ == "__main__":
  import utils
  import finetune
  from torch.utils.data import DataLoader

  # update the paths to the classifier and the finetuned vision_tower
  classifier_path = "./classifier-epoch-1-lr-0.0001.pth"
  vision_tower_path = "./vision_tower-epoch-1-lr-0.0001"

  classifier = finetune.load_classifier(classifier_path)
  vision_tower_instance = finetune.load_existing_vision(vision_tower_path)

  # annotations file: path to csv file containing the image path, and the ground truth, the 14 pathologies annotations
  # img_dir is concatenated with the file path in annotations file to obtain the final image path
  validation_data = finetune.CustomImageDataset(
      annotations_file="/MIMIC-CXR/processed_data/processed_mimic-cxr-2.0.0-chexpert_validate.csv",
      img_dir="/physionet.org/files/mimic-cxr-jpg/2.1.0",
  )
  validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)

  y_true, y_pred = finetune.test_validation_set(vision_tower_instance, classifier, validation_dataloader)

  y_pred = utils.get_classification(y_pred).cpu()
  y_true = utils.flatten(y_true).cpu()

  # evaluate the performance of the model
  from sklearn import metrics
  accuracy_score = metrics.accuracy_score(y_pred=y_pred, y_true=y_true)
  recall_score = metrics.recall_score(y_pred=y_pred, y_true=y_true)
  precision_score = metrics.precision_score(y_pred=y_pred, y_true=y_true)
  f1_score = metrics.f1_score(y_pred=y_pred, y_true=y_true)

  print("accuracy score:", accuracy_score)
  print("precision score:", precision_score)
  print("recall score:", recall_score)
  print("f1 score:", f1_score)
