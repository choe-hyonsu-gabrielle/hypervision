import os
import torch
import pytorch_lightning as pl
from hypervision.session import HypervisionSession
from supervision.data.datamodule import BaselineCSVsDataModule
from supervision.modeling.bert_model_kit import BertClassifierConfig, BertClassifierModel


# huggingface/tokenizers: The current process just got forked, after parallelism has already been used.
# Disabling parallelism to avoid deadlocks... To disable this warning, you can either:
# - Avoid using `tokenizers` before the fork if possible
# - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,"

# when you are using a CUDA device ('NVIDIA A100-SXM4-40GB') that has tensor cores.
torch.set_float32_matmul_precision('high')


# HypervisionSession class is just grid search hyperparameter tuner class.
# It basically generates a set of training configurations and make it easier to manage multiple supervision sessions
class HyperParameterTuning(HypervisionSession):
    def __init__(self, session_name: str):
        super().__init__(session_name)
        self.model_names: list[str] = [                 # list of pretrained_model_name_or_paths
            'klue/bert-base',
        ]
        self.dataset_params: list[dict] = [             # list of dataset paths.
            {
                'name': 'baseline.v1',
                'num_classes': 2,
                'train': '../supervision/data/corpus/baselines/*.csv',
                'validation': None,
                'test': None,
                'train_validation_split': {'train': 0.9, 'validation': 0.1}
            },
        ]
        self.hyperparams: dict = {                       # hyperparameters to be distributed for each training session
            'batch_size': [64, 128],
            'learning_rate': [1e-5, 5e-5],
        }
        self.callback_params: dict = {
            'monitor': 'valid_loss',
            'mode': 'min',
            'patience': 5,
        }
        self.trainer_params: dict = {
            'max_epochs': 20,
            'devices': [0, 1, 2, 3],                        # gpu device ids for pl.Trainer
            'strategy': 'ddp_find_unused_parameters_true',  # when you are using BERT
            # 'strategy': 'auto',
        }
        self.random_seed: int = 42
        self.checkpoints_dir: str = 'checkpoints'           # logging & checkpoint directory
        self.logging_dir: str = "logs"


if __name__ == '__main__':
    # HypervisionSession is a kind of data class that holds configurations for each training events.
    hypervisor = HyperParameterTuning(session_name='TEST')

    # This is just a grid-search hyperparameter tuning loop. You get sub-sessions here ...
    for session in hypervisor.supervision_sessions:
        # ... and start a sub-session explicitly.
        session.initialize()

        # This is an independent model config. All args you need are stored in `session`.
        config = BertClassifierConfig(
            pretrained_model_name_or_path=session.pretrained_model_name_or_path,
            num_classes=session.num_classes,
            batch_size=session.batch_size,
            learning_rate=session.learning_rate
        )

        # This is an independent LightningModule model.
        model = BertClassifierModel(config=config)

        # This is an independent DataModule.
        datamodule = HateSpeechBaselineCSVsDataModule(
            train_dir=session.train_dir,
            validation_dir=session.validation_dir,
            test_dir=session.test_dir,
            batch_size=session.batch_size,
            train_validation_ratio=session.train_validation_ratio,
            random_seed=hypervisor.random_seed
        )

        # This is a Lightning Trainer.
        trainer = pl.Trainer(
            max_epochs=session.trainer_params['max_epochs'],
            accelerator='auto',
            devices=session.trainer_params['devices'],
            strategy=session.trainer_params['strategy'],
            callbacks=session.callbacks,
            logger=session.tensorboard_logger
        )

        # LET'S GO!!!
        trainer.fit(model=model, datamodule=datamodule)

        # ... and close the sub-session.
        session.terminate(save_description=True)

    # checkpoints are automatically saved by callback.
    print(hypervisor.best_scored_session)
