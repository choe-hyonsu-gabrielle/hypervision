import csv
from tqdm import tqdm


def load_csv(filename: str, encoding: str = 'utf-8'):
    """ csv 파일 하나를 불러 오고 데이터를 반환
    :param filename: filename
    :param encoding: encoding option for open()
    :return: json data
    """
    with open(filename, encoding=encoding) as fp:
        csv_items = csv.DictReader(fp)
        fields = csv_items.fieldnames
        return [{field: row[field] for field in fields} for row in csv_items]


def load_csvs(filepaths: list[str], encoding: str = 'utf-8', flatten: bool = False) -> list:
    """ 여러 개의 csv 파일을 불러 오고 데이터를 list 형태로 반환
    :param filepaths: filepath list (ex. ["./1.json", "./2.json", ...]
    :param encoding: encoding option for open()
    :param flatten: 불러온 모든 데이터를 단일 list로 반환
    :return: list of json data
    """
    results = []
    for filepath in tqdm(filepaths, desc=f'- loading {len(filepaths)} files'):
        assert filepath.endswith('.csv'), filepath
        if not flatten:
            results.append(load_csv(filename=filepath, encoding=encoding))
        else:
            results.extend(load_csv(filename=filepath, encoding=encoding))
    return results
  
