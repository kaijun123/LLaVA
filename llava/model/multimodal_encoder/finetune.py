if __name__ == "__main__":
    import utils
    from torch.utils.data import DataLoader

    # Finetune the clip encoder
    vision_tower = utils.build_vision_tower(utils.ModelArguments())
    classifier = utils.CLIPDiseaseClassifier()
    print("vision_tower:", vision_tower)
    print("classifier:", classifier)

    training_data = utils.CustomImageDataset(
        annotations_file="/MIMIC-CXR/processed_data/processed_mimic-cxr-2.0.0-chexpert_train.csv",
        img_dir="/physionet.org/files/mimic-cxr-jpg/2.1.0",
    )
    validation_data = utils.CustomImageDataset(
        annotations_file="/MIMIC-CXR/processed_data/processed_mimic-cxr-2.0.0-chexpert_validate.csv",
        img_dir="/physionet.org/files/mimic-cxr-jpg/2.1.0",
    )

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)

    # finetune the clip encoder on the training dataset 
    utils.train_clip_classifier(
        vision_tower,
        classifier,
        train_loader=train_dataloader,
        val_loader=validation_dataloader,
        output_dir=".",
        epochs=1,
    )