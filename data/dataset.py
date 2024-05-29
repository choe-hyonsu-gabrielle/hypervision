from torch.utils.data import Dataset
from utils.path import filepath_resolution
from utils.json import load_jsons
from utils.csv import load_csvs


class DatasetBase(Dataset):
    def __init__(self, filenames: list[str]):
        self.filenames = filepath_resolution(filenames)
        print(f'[{self.__class__.__name__}] initializing Dataset from: {self.filenames}')
        self.data: list[dict] = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item: dict = self.data[idx]
        return item


class MultipleJSONsDataset(DatasetBase):
    def __init__(self, filenames: list[str]):
        super().__init__(filenames)
        self.data = load_jsons(self.filenames, encoding='utf-8', flatten=True)


class MultipleCSVsDataset(DatasetBase):
    def __init__(self, filenames: list[str]):
        super().__init__(filenames)
        self.data = load_csvs(self.filenames, encoding='utf-8', flatten=True)


if __name__ == '__main__':
    csvs = MultipleCSVsDataset(filenames=['corpus/baselines/train/*.csv'])
    print(len(csvs))

    for i, d in enumerate(csvs):
        print(d)
        if i == 9:
            break
