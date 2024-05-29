import glob
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from data.datamodule import BaselineCSVsDataModule
from supervision.modeling.bert_classifier_kit import BertClassifierConfig, BertClassifierModel


# This is just an example of single training event with Pytorch Lightning.
if __name__ == '__main__':
    # This is an independent model config.
    config = BertClassifierConfig(
        pretrained_model_name_or_path='klue/bert-base',
        num_classes=2,
        batch_size=64,
        learning_rate=1e-5
    )

    # This is an independent LightningModule model.
    model = BertClassifierModel(config=config)

    # This is an independent DataModule.
    datamodule = BaselineCSVsDataModule(
        train_dir=glob.glob('../data/corpus/baselines/train/*.csv'),
        batch_size=64,
        train_validation_ratio={'train': 0.9, 'validation': 0.1},
        random_seed=42
    )

    # Now you have callbacks and Tensorboard logger.
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath='checkpoints/',
            filename='YOUR_AWESOME_MODEL.epoch.{epoch}-vloss.{valid_loss:.5f}',
            monitor='valid_loss',
            mode='min',
            auto_insert_metric_name=False,
            verbose=True
        ),
        pl.callbacks.EarlyStopping(
            monitor='valid_loss',
            mode='min',
            min_delta=0.00001,
            patience=5,
            verbose=True,
            check_finite=True
        )
    ]
    tensorboard_logger = TensorBoardLogger(
        save_dir='tensorboard_logs',
        name='logger',
        version='YOUR_AWESOME_MODEL_NAME'
    )

    # This is a Lightning Trainer.
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='auto',
        devices=[0, 1, 2, 3],
        strategy='auto',
        callbacks=callbacks,
        logger=tensorboard_logger
    )

    # LET'S GO!!!
    trainer.fit(model=model, datamodule=datamodule)
