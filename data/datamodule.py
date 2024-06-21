import random
from torch import Generator
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from data.dataset import MultipleCSVsDataset
from data.collator import BaselineCSVsCollator


class DataModuleBase(pl.LightningDataModule):
    def __init__(self, train_dir: list[str] = None, validation_dir: list[str] = None, test_dir: list[str] = None,
                 prediction_dir: list[str] = None, batch_size: int = 64, train_validation_ratio: dict = None,
                 random_seed: int = 42, num_workers: int = 8):
        super().__init__()
        self.train_dir = train_dir
        self.validation_dir = validation_dir
        self.test_dir = test_dir
        self.prediction_dir = prediction_dir

        # hyperparameters
        self.batch_size = batch_size
        self.train_validation_ratio = train_validation_ratio
        self.random_seed = random_seed
        self.num_workers = num_workers

        # dataloaders
        self.train = None
        self.validation = None
        self.test = None
        self.predict = None

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        assert self.train
        print(f'[{self.__class__.__name__}] train {len(self.train)} batches')
        return self.train

    def val_dataloader(self) -> EVAL_DATALOADERS:
        assert self.validation
        print(f'[{self.__class__.__name__}] validation {len(self.validation)} batches')
        return self.validation

    def test_dataloader(self) -> EVAL_DATALOADERS:
        assert self.test
        print(f'[{self.__class__.__name__}] test {len(self.test)} batches')
        return self.test

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        assert self.predict
        print(f'[{self.__class__.__name__}] test {len(self.predict)} batches')
        return self.predict


class BaselineCSVsDataModule(DataModuleBase):
    def __init__(self, train_dir: list[str] = None, validation_dir: list[str] = None, test_dir: list[str] = None,
                 prediction_dir: list[str] = None, batch_size: int = 64, train_validation_ratio: dict = None,
                 random_seed: int = 42, num_workers: int = 8):
        super().__init__(train_dir, validation_dir, test_dir, prediction_dir, batch_size, train_validation_ratio,
                         random_seed, num_workers)
        self.collator = BaselineCSVsCollator(label_map={'positive': 1, 'negative': 0})

    def setup(self, stage: str) -> None:
        print(f'[{self.__class__.__name__}] setup(`stage`)="{stage}"')
        if self.random_seed:
            random.seed(self.random_seed)
        if stage == 'fit':
            assert self.train_dir
            train_dataset = MultipleCSVsDataset(filenames=self.train_dir)
            if self.train_validation_ratio and not self.validation_dir:
                print(f'[{self.__class__.__name__}] split train & validation dataset with {self.train_validation_ratio}')
                train_dataset, validation_dataset = random_split(
                    dataset=train_dataset,
                    lengths=(self.train_validation_ratio['train'], self.train_validation_ratio['validation']),
                    generator=Generator().manual_seed(42)
                )
            elif self.validation_dir:
                print(f'[{self.__class__.__name__}] loading validation dataset regardless of {self.train_validation_ratio}')
                validation_dataset = MultipleCSVsDataset(filenames=self.validation_dir)
            else:
                raise ValueError(f"[{self.__class__.__name__}] `validation_dir` or `train_validation_ratio` must be provided.")
            self.train = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.collator
            )
            self.validation = DataLoader(
                dataset=validation_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collator
            )
        elif stage == 'validate':
            assert self.validation_dir
            validation_dataset = MultipleCSVsDataset(filenames=self.validation_dir)
            self.validation = DataLoader(
                dataset=validation_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collator
            )
        elif stage == 'test':
            assert self.test_dir
            self.test = DataLoader(
                dataset=MultipleCSVsDataset(filenames=self.test_dir),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collator
            )
        elif stage == 'predict':
            assert self.prediction_dir
            self.test = DataLoader(
                dataset=MultipleCSVsDataset(filenames=self.prediction_dir),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collator
            )
        else:
            raise ValueError(f"[{self.__class__.__name__}] `stage` must be either 'fit', 'validate', 'test', or 'predict'")


if __name__ == '__main__':
    dm = BaselineCSVsDataModule(
        train_dir=[
            'corpus/baselines/train/1.csv',
            'corpus/baselines/train/2.csv',
            'corpus/baselines/train/3.csv'
        ],
        train_validation_ratio={'train': 0.9, 'validation': 0.1},
        batch_size=4
    )
    dm.setup('fit')

    for i, x in enumerate(dm.val_dataloader()):
        for k, v in x.items():
            print(f'- "{k}": {v}')
        print()
        if i == 9:
            break
