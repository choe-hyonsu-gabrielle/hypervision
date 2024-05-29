import os
import glob
import warnings
from sklearn.metrics import classification_report
from data.datamodule import BaselineCSVsDataModule
from supervision.modeling.bert_model_kit import BertClassifierConfig, BertClassifierModel

warnings.filterwarnings(action='ignore')


if __name__ == '__main__':
    config = BertClassifierConfig(pretrained_model_name_or_path='klue/bert-base', num_classes=2)
    # It will not load immediately bare pretrained model from HuggingFace Hub until requested.

    model = BertClassifierModel.load_from_checkpoint(
        checkpoint_path='../hypervision/checkpoints/YOUR_AWESOME_CHECKPOINT.ckpt',
        map_location={'cuda:0': 'cuda:0'},
        **{'config': config}
    )

    datamodule = BaselineCSVsDataModule(test_dir=['../supervision/data/corpus/baselines/test.csv'], batch_size=32)
    datamodule.setup('test')

    ground_truth = []
    model_prediction = []

    for batch in datamodule.test_dataloader():
        ground_truth.extend(batch['labels'])
        model_prediction.extend(model.predict(batch).cpu().tolist())

    report = classification_report(
        y_true=ground_truth,
        y_pred=model_prediction,
        labels=[0, 1],
        target_names=['negative', 'positive'],
        digits=4,
        zero_division=0.0
    )
      
    print(report)
