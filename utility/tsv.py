from tqdm import tqdm


def load_tsv(filename: str, encoding: str = 'utf-8', first_as_column_names: bool = True, map_columns: list = None):
    """ tsv 파일 하나를 불러 오고 데이터를 반환
    :param filename: filename
    :param encoding: encoding option for open()
    :param first_as_column_names: take first line as column names
    :param map_columns: replace or use column names as provided by user
    :return: json data
    """
    with open(filename, encoding=encoding) as fp:
        column_names = None
        data = []
        for i, line in enumerate(fp.readlines()):
            if first_as_column_names:
                if i == 0:
                    column_names = line.strip().split('\t')
                    continue
            values = line.strip().split('\t')
            if column_names:
                assert len(values) == len(column_names)
                if map_columns:
                    assert len(map_columns) == len(column_names)
                    data.append({k: v for k, v in zip(map_columns, values)})
                else:
                    data.append({k: v for k, v in zip(column_names, values)})
            else:
                if map_columns:
                    assert len(map_columns) == len(values)
                    data.append({k: v for k, v in zip(map_columns, values)})
                else:
                    data.append({k: values[k] for k in range(len(values))})
        return data


def load_tsvs(filepaths: list[str], encoding: str = 'utf-8', flatten: bool = False, first_as_column_names: bool = True, map_columns: list = None) -> list:
    """ 여러 개의 csv 파일을 불러 오고 데이터를 list 형태로 반환
    :param filepaths: filepath list (ex. ["./1.json", "./2.json", ...]
    :param encoding: encoding option for open()
    :param flatten: 불러온 모든 데이터를 단일 list로 반환
    :param first_as_column_names: take first line as column names
    :param map_columns: replace or use column names as provided by user
    :return: list of json data
    """
    results = []
    for filepath in tqdm(filepaths, desc=f'- loading {len(filepaths)} files'):
        assert filepath.endswith('.tsv'), filepath
        if not flatten:
            results.append(
                load_tsv(
                    filename=filepath,
                    encoding=encoding,
                    first_as_column_names=first_as_column_names,
                    map_columns=map_columns
                )
            )
        else:
            results.extend(
                load_tsv(
                    filename=filepath,
                    encoding=encoding,
                    first_as_column_names=first_as_column_names,
                    map_columns=map_columns
                )
            )
    return results
