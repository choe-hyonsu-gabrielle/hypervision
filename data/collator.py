class BaselineCSVsCollator:
    def __init__(self, label_map: dict = None):
        self.label_map = label_map if label_map else {'positive': 1, 'negative': 0}

    def __call__(self, items, *args, **kwargs):
        collated = dict(uids=[], targets=[], texts=[], labels=[], annotations=[])
        for item in items:
            uid = item['uid'] if 'uid' in item else None
            annotation = item['annotation']
            label = self.label_map[annotation]
            target = [0] * len(self.label_map)
            target[label] = 1
            text = item['text']
            collated['uids'].append(uid)
            collated['targets'].append(target)
            collated['labels'].append(label)
            collated['texts'].append(text)
            collated['annotations'].append(annotation)
        return collated
