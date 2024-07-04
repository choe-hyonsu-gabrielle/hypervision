import os
import glob
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from data.datamodule import BaselineCSVsDataModule
from modeling.sentence_classifier_kit import SentenceClassificationConfig, SentenceClassificationModel


torch.set_printoptions(edgeitems=10, linewidth=200)
torch.set_num_threads(4)

# when you are using a CUDA device ('NVIDIA A100-SXM4-40GB') that has Tensor Cores.
torch.set_float32_matmul_precision('high')

# huggingface/tokenizers: The current process just got forked, after parallelism has already been used.
# Disabling parallelism to avoid deadlocks... To disable this warning, you can either:
# 	- Avoid using `tokenizers` before the fork if possible
# 	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


# This is just an example of single training event with Pytorch Lightning.
if __name__ == '__main__':
    # This is an independent model config.
    config = SentenceClassificationConfig(
        pretrained_model_name_or_path='klue/bert-base',
        num_classes=2,
        batch_size=64,
        learning_rate=1e-5,
        pooling_strategy='cls',
        max_seq_length='max'
    )

    # This is an independent LightningModule model.
    model = SentenceClassificationModel(config=config)

    # This is an independent DataModule.
    datamodule = BaselineCSVsDataModule(
        train_dir=glob.glob('../data/corpus/baselines/train/*.csv'),
        test_dir=glob.glob('../data/corpus/baselines/test/*.csv'),
        train_validation_ratio={'train': 0.9, 'validation': 0.1},
        batch_size=64,
        random_seed=42
    )

    # Now you have callbacks and Tensorboard logger.
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath='checkpoints/',
            filename='YOUR_AWESOME_MODEL_NAME.epoch.{epoch}-vloss.{valid/loss:.5f}',
            monitor='valid/loss',
            mode='min',
            auto_insert_metric_name=False,
            save_last=True,
            save_top_k=3,
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
        name='tensorboard_logger',
        version='YOUR_AWESOME_MODEL_NAME'
    )

    # This is a Lightning Trainer.
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='auto',
        devices=[0, 1, 2, 3],
        strategy='auto',  # 'ddp_find_unused_parameters_true' when using BERT models
        callbacks=callbacks,
        logger=tensorboard_logger,
        log_every_n_steps=10
    )

    # LET'S GO!!!
    model.train()
    trainer.fit(model=model, datamodule=datamodule)
