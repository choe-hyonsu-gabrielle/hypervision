# hypervision

***
### Requirements
```shell
pip install transformers datasets lightnings scikit-learn tensorboard tensorboardX
```

***
### About `supervision`

- `supervision` contains base classes for model architecture and training. `supervision.modeling` has basic model kits,
which comprises a couple of model (a subclass of `model.LightningModuleBase`) and its configuration
(a subclass of `config.ModelConfigBase`)
 
- It is very encouraged to make your own model kits under `supervision.modeling`.
You can easily place activation, objective, learning rate scheduler and whatever you want to put in your model
in the configuration class. (See `BertClassifierConfig` as an example.)

- Once you defined your original model class (ex. `BertClassifierModel`) and model config class (ex. `BertClassifierConfig`),
you can simply initialize a model object, which is actually `pl.LightningModule` at the core, by passing
a model config object that you've just implemented.

- Please be noticed that `config.ModelConfigBase` holds all of pretrained language model (`AutoModel`) and 
tokenizer (`AutoTokenizer`) from `transformers` at first. Then `model.LightningModuleBase` will automatically 
load the pretrained artifacts from model config object you've just passed to.

```python
from supervision.modeling.bert_model_kit import BertClassifierConfig, BertClassifierModel

config = BertClassifierConfig(
    pretrained_model_name_or_path='klue/bert-base',  # AutoTokenizer & AutoModel are prepared to be fed to model later.
    num_classes=2,
    batch_size=32,
    learning_rate=1e-5
)

model = BertClassifierModel(config)  # model initiated with pretrained artifacts from config.
```
- `supervision.data` has all codes for training and evaluation data. You need to define custom `DatasetBase`, 
(a subclass of `Dataset` from `torch`) `DataModuleBase` (a subclass of `pl.LightningDataModule`) and
custom collator function. (See `collator.BaselineCSVsCollator` as an example.)

#### Basic training with `supervision`
- You can see detailed, working code example of single training event at `supervision/training.py`.
```python
import pytorch_lightning as pl
from supervision.modeling.bert_model_kit import BertClassifierConfig, BertClassifierModel
from supervision.data.datamodule import BaselineCSVsDataModule
 
# loading model & data
config = BertClassifierConfig(**model_params)                   # custom model config
model = BertClassifierModel(config)                             # pl.LightningModule
datamodule = BaselineCSVsDataModule(**datamodule_params)        # pl.LightningDataModule
 
# trainer
trainer = pl.Trainer(**trainer_params)
trainer.fit(model, datamodule)
```

***
### About `hypervision`

- `hypervision.session` has two classes for hyperparameter tuning. `HypervisionSession` is a supervisor (or *'hypervisor'*)
of all subordinate supervised learning sessions (`SupervisionSession`) those who has a distinct set of hyperparameters
required by `config`, `model`, `datamodule` and `trainer` at every individual session.

- You can easily set up a hyperparameter tuning loop based on grid search method. Just put in the supervision codes
under the `for` loop which is motivated by `HypervisionSession.supervision_sessions`.

#### Basic hyperparameter tuning with `hypervision`
- You can see detailed, working code example of hyperparameter tuning at `hypervision/training.py`.
```python
import pytorch_lightning as pl
from hypervision.session import HypervisionSession
from supervision.modeling.bert_model_kit import BertClassifierConfig, BertClassifierModel
from supervision.data.datamodule import BaselineCSVsDataModule

hypervisor = HypervisionSession(session_name='DEMO')
 
for session in hypervisor.supervision_sessions:
    # begin of supervision session: it will internally set up callbacks and tensorboard logger.
    session.initialize()
     
    # put your supervision codes under the loop.
    config = BertClassifierConfig(**session.model_params)                 # custom model config
    model = BertClassifierModel(config)                                   # pl.LightningModule
    datamodule = BaselineCSVsDataModule(**session.datamodule_params)      # pl.LightningDataModule
     
    trainer = pl.Trainer(**session.trainer_params)
    trainer.fit(model, datamodule)
 
    # end of supervision session: wrapping up with best score and checkpoint are registered.
    session.terminate()
 
best_model = hypervisor.best_scored_session  # end of hyperparameter tuning loop.
```
