import os
import json
import datetime
from typing import Optional, Literal
from itertools import product
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from utility.path import filepath_resolution
from utility.logging import get_event_logger


class HypervisionSession:
    """
    HypervisionSession class is just grid search hyperparameter tuner class.
    It basically generates a set of training configurations and make it easier to manage multiple supervision sessions
    """
    def __init__(self, session_name: str):
        self.session_name = session_name                # it will be used as an alias
        self.model_names: list[str] = [                 # list of pretrained_model_name_or_paths
            'klue/bert-base',
            'klue/roberta-large'
        ]
        self.dataset_params: list[dict] = [             # list of dataset paths.
            {
                'name': 'baseline.v1',
                'num_classes': 2,
                'train': '../data/corpus/train/*.csv',
                'validation': None,
                'test': None,
                'train_validation_split': {'train': 0.9, 'validation': 0.1}
            },
        ]
        self.hyperparams: dict = {                      # hyperparameters to be distributed for each training session
            'batch_size': [16, 32, 64, 128],
            'learning_rate': [5e-6, 1e-5, 5e-5, 1e-4],
            'pooling_strategy': ['cls', 'mean', 'max', 'pooler_output']
        }
        self.callback_params: dict = {
            'monitor': 'valid/loss',
            'mode': 'min',
            'patience': 5,
            'min_delta': 0.00001
        }
        self.trainer_params: dict = {
            'max_epochs': 20,
            'devices': [0, 1, 2, 3],  # gpu device ids for pl.Trainer
            'strategy': 'auto'  # or 'ddp_find_unused_parameters_true'
            'log_every_n_steps': 100
        }

        # model restructuring
        self.additional_special_tokens: Optional[list[str]] = None

        # constants
        self.random_seed: int = 42
        self.checkpoints_dir: str = 'checkpoints'  # logging & checkpoint directory
        self.logging_dir: str = "logs"

        # event logging
        os.makedirs(self.logging_dir, exist_ok=True)
        self.event_logger = get_event_logger(
            logger_name=self.session_name,
            filename=f'{self.logging_dir}/{self.session_name}-events.log'
        )

        # supervision sessions
        self._supervision_sessions = []
        self._reports = []

        # sanity checks
        self._sanity_checks()

    def _sanity_checks(self):
        # any sanity checking functions you want
        pass

    def _individualize(self, model_name, dataset_param, batch_size, learning_rate, pooling_strategy):
        supervision_session_name = '.'.join([
            self.session_name,                          # using hypervisor session name as a prefix
            model_name,                                 # any '/' in model name will be replaced with '-'
            dataset_param['name'],                      # dataset_name
            f'lr={learning_rate}.bs={batch_size}.ps={pooling_strategy}'  # learning_rate & batch_size as identifier
        ]).replace('/', '-')
        return SupervisionSession(
            hypervisor=self,
            session_name=supervision_session_name,
            pretrained_model_name_or_path=model_name,
            batch_size=batch_size,
            learning_rate=learning_rate,
            pooling_strategy=pooling_strategy,
            num_classes=dataset_param['num_classes'],
            train_dir=dataset_param['train'],
            validation_dir=dataset_param['validation'],
            test_dir=dataset_param['test'],
            train_validation_ratio=dataset_param['train_validation_split'],
            callback_params=self.callback_params,
            trainer_params=self.trainer_params,
            additional_special_tokens=self.additional_special_tokens
        )

    @property
    def supervision_sessions(self) -> list['SupervisionSession']:
        if not self._supervision_sessions:
            for model_name, dataset_param in product(self.model_names, self.dataset_params):
                production_args = [
                    self.hyperparams['batch_size'],
                    self.hyperparams['learning_rate'],
                    self.hyperparams['pooling_strategy']
                ]
                for batch_size, learning_rate, pooling_strategy in product(*production_args):
                    supervision_session = self._individualize(model_name, dataset_param, batch_size, learning_rate, pooling_strategy)
                    self._supervision_sessions.append(supervision_session)
        return self._supervision_sessions

    @property
    def best_scored_session(self) -> [dict, None]:
        valid_scored_sessions = [r for r in self._reports if r['best_model_score']]
        if not valid_scored_sessions:
            return None
        if self.callback_params['mode'] == 'min':
            return sorted(valid_scored_sessions, key=lambda x: x['best_model_score'])[0]
        elif self.callback_params['mode'] == 'max':
            return sorted(valid_scored_sessions, key=lambda x: x['best_model_score'])[-1]
        else:
            raise AttributeError('it returns best one correspond to whether callback_params["mode"] is `min` or `max`.')

    def reception(self, session_description: dict):
        # it is used by a supervision session to report score and checkpoint path when it is finished
        self._reports.append(session_description)


class SupervisionSession:
    def __init__(self, hypervisor: HypervisionSession, session_name: str, pretrained_model_name_or_path: str,
                 batch_size: int, learning_rate: float, pooling_strategy: Literal['cls', 'mean', 'max', 'pooler_output'],
                 num_classes: int, train_dir: (str, list[str]), validation_dir: (str, list[str]), test_dir: (str, list[str]),
                 train_validation_ratio: dict, callback_params: dict, trainer_params: dict, additional_special_tokens: list[str]):
        self.hypervisor = hypervisor
        self.session_name = session_name
        self.version = 'NOT_DETERMINED_YET'
        self.trial_no = None

        # personalized hyperparameters
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pooling_strategy = pooling_strategy
        self.additional_special_tokens = additional_special_tokens

        # dataset configurations
        self.num_classes = num_classes
        self.train_dir: list = filepath_resolution(train_dir, absolute=True)
        self.validation_dir: list = filepath_resolution(validation_dir, absolute=True)
        self.test_dir: list = filepath_resolution(test_dir, absolute=True)
        self.train_validation_ratio = train_validation_ratio

        # personalized functions
        self.callback_params = callback_params
        self.callbacks = None
        self.trainer_params = trainer_params
        self.tensorboard_logger = None

        # privates
        self.started_at = None
        self.ended_at = None
        self.elapsed_time = None
        self.notes = dict()

    @property
    def model_params(self):
        return dict(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            num_classes=self.num_classes,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            pooling_strategy=self.pooling_strategy
        )

    @property
    def datamodule_params(self):
        return dict(
            train_dir=self.train_dir,
            validation_dir=self.validation_dir,
            test_dir=self.test_dir,
            batch_size=self.batch_size,
            train_validation_ratio=self.train_validation_ratio,
            random_seed=self.hypervisor.random_seed,
        )

    def __repr__(self):
        contents = {k: v for k, v in self.__dict__.items() if k in ('session_name', 'version', 'trial_no',)}
        return f'[{self.__class__.__name__}] {contents}'

    def __enter__(self):
        self.initiate()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

    def initiate(self):
        # initiate version, loggers and callbacks
        self.started_at = datetime.datetime.now()
        self.version = self.started_at.strftime("v%y%m%d-%p%H%M")
        self.tensorboard_logger = TensorBoardLogger(
            save_dir=self.hypervisor.logging_dir,
            name=self.session_name,
            version=self.version
        )
        self.callbacks = [
            ModelCheckpoint(
                dirpath=self.hypervisor.checkpoints_dir,
                filename='.'.join([self.session_name, self.version, 'epoch.{epoch}-vloss.{valid/loss:.5f}']),
                monitor=self.callback_params['monitor'],
                mode=self.callback_params['mode'],
                auto_insert_metric_name=False,
                verbose=True
            ),
            EarlyStopping(
                monitor=self.callback_params['monitor'],
                mode=self.callback_params['mode'],
                min_delta=self.callback_params['min_delta'],
                patience=self.callback_params['patience'],
                verbose=True,
                check_finite=True
            )
        ]
        self.trainer_params['callbacks'] = self.callbacks
        self.trainer_params['logger'] = self.tensorboard_logger
        self.trial_no = f'{self.hypervisor.supervision_sessions.index(self) + 1}/{len(self.hypervisor.supervision_sessions)}'
        self.hypervisor.event_logger.info(
            f'[{self.hypervisor.__class__.__name__}] initiate `{self.session_name}` (trial {self.trial_no}): {str(self)}'
        )

    @property
    def checkpoint_callback(self) -> Optional[ModelCheckpoint]:
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                return callback
        return None

    def description(self, as_dict: bool = False):
        desc = dict(hypervision_configuration=dict(), supervision_configuration=dict())
        for k, v in self.hypervisor.__dict__.items():
            if not k.startswith('_'):
                desc['hypervision_configuration'][k] = v
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                desc['supervision_configuration'][k] = v
        # to avoid handling objects that cannot be serialized, we make json string with `default=str` once.
        serialized = json.dumps(desc, ensure_ascii=False, indent=4, default=str)
        if as_dict:
            return json.loads(serialized)
        return serialized

    def terminate(self, save_description: bool = True):
        self.ended_at = datetime.datetime.now()
        self.elapsed_time = str(self.ended_at - self.started_at)
        checkpoint_callback = self.checkpoint_callback    # callbacks.model_checkpoint.ModelCheckpoint
        desc = dict(
            best_model_path=checkpoint_callback.best_model_path,
            best_model_score=checkpoint_callback.best_model_score,
            elapsed_time=self.elapsed_time,
            notes=self.notes
        )
        session_description = self.description(as_dict=True)
        desc.update(session_description)
        if save_description:
            with open(f'{self.hypervisor.logging_dir}/{self.session_name}/session_desc.json', 'w', encoding='utf-8') as fp:
                json.dump(self.description(as_dict=True), fp, ensure_ascii=False, indent=4)
        self.hypervisor.reception(desc)
        self.hypervisor.event_logger.info(
            f'[{self.hypervisor.__class__.__name__}] terminate `{self.session_name}` (trial {self.trial_no}): {str(self)}\n'
            f'[{self.hypervisor.__class__.__name__}] best_model_path: {checkpoint_callback.best_model_path}\n'
            f'[{self.hypervisor.__class__.__name__}] best_model_score: {checkpoint_callback.best_model_score}\n'
            f'[{self.hypervisor.__class__.__name__}] elapsed_time: {self.elapsed_time}\n'
            f'[{self.hypervisor.__class__.__name__}] notes: {self.notes}'
        )


if __name__ == '__main__':
    hypervisor = HypervisionSession(session_name='DEMO')

    for session in hypervisor.supervision_sessions:
        session.initiate()
        # do something training a model with a supervision session
        session.terminate()
