import os
import torch
import pytorch_lightning as pl
from data.datamodule import BaselineCSVsDataModule
from hypervision.session import HypervisionSession
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


# HypervisionSession class is just grid search hyperparameter tuner class.
# It basically generates a set of training configurations and make it easier to manage multiple supervision sessions
class HyperParameterTuning(HypervisionSession):
    def __init__(self, session_name: str):
        super().__init__(session_name)
        self.model_names: list[str] = [                 # list of pretrained_model_name_or_paths
            'klue/bert-base',
            'klue/roberta-large'
        ]
        self.dataset_params: list[dict] = [             # list of dataset paths.
            {
                'name': 'baseline.v1',
                'num_classes': 2,
                'train': ['../data/corpus/train/*.csv'],
                'validation': None,
                'test': ['../data/corpus/test/*.csv'],
                'train_validation_split': {'train': 0.9, 'validation': 0.1}
            },
        ]
        self.hyperparams: dict = {                       # hyperparameters to be distributed for each training session
            'batch_size': [64, 128],
            'learning_rate': [1e-5, 5e-5],
            'pooling_strategy': ['cls'],                 # ['cls', 'mean', 'max', 'pooler_output']
            'max_seq_length': ['max']                    # `None` or 'max' means maximum sequence length
        }
        self.callback_params: dict = {
            'monitor': 'valid/loss',
            'mode': 'min',
            'patience': 5,
            'min_delta': 0.00001
        }
        self.trainer_params: dict = {
            'max_epochs': 20,
            'accelerator': 'auto',
            'devices': [0, 1, 2, 3],                        # gpu device ids for pl.Trainer
            'strategy': 'ddp_find_unused_parameters_true',  # when you are using BERT
            # 'strategy': 'auto',
            'log_every_n_steps': 50
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
        session.initiate()

        # This is an independent model config. All args you need are stored in `session`.
        config = SentenceClassificationConfig(**session.model_params)

        # This is an independent LightningModule model.
        model = SentenceClassificationModel(config)

        # This is an independent DataModule.
        datamodule = BaselineCSVsDataModule(**session.datamodule_params)

        # This is a Lightning Trainer.
        trainer = pl.Trainer(**session.trainer_params)

        # check if you want to full fine-tuning or not before start
        model.train()
        
        # LET'S GO!!!
        trainer.fit(model=model, datamodule=datamodule)

        # you can append a test set loss on the session log
        session.notes['trainer.test'] = trainer.test(model=model, datamodule=datamodule)

        # ... and close the sub-session.
        session.terminate()

    # checkpoints are automatically saved by callback.
    print(hypervisor.best_scored_session)
