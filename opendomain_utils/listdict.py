import pandas as pd
import csv
import os
from collections import OrderedDict


class ListDict:
    def __init__(self, data=None, **kwargs):
        if data is None:
            data = []
        self.data = data
        self.kwargs = kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key: (int, slice, str, tuple, list)):
        if isinstance(key, str):
            return [p[key] for p in self.data]
        elif isinstance(key, int):
            return self.data[key]
        elif isinstance(key, slice):
            return self.__class__(data=self.data[key], **self.kwargs)
        elif isinstance(key, (tuple, list)):
            if len(key) == 1:
                return self[key[0]]
            records = []
            for key_ in key:
                records.append(self[key_])
            if isinstance(records[-1], (dict, OrderedDict)):
                return self.__class__(data=records, **self.kwargs)
            else:
                return list(zip(*records))
        else:
            raise TypeError('Key must be str or list')

    def __str__(self):
        s = []
        for i in self.data:
            s.append(str(i))
        return '\n'.join(s)

    @property
    def header(self):
        if len(self.data) > 0:
            return list(self.data[0].keys())
        else:
            return None

    def get(self, key, default=None):
        try:
            return self[key]
        except:
            return default

    def append(self, data):
        if isinstance(data, ListDict):
            assert len(data) == 1
            data = data.data[0]
        if isinstance(data, (dict, OrderedDict)):
            self.data.append(data)
        else:
            raise TypeError('Method append does support for type {}'.format(type(data)))

    def extend(self, data):
        if isinstance(data, ListDict):
            data = data.data
        if isinstance(data, list):
            self.data.extend(data)
        else:
            raise TypeError('Method extend does support for type {}'.format(type(data)))

    def update(self, data):
        if isinstance(data, ListDict):
            if len(data) == 1:
                self.append(data)
            else:
                self.extend(data)
        elif isinstance(data, (dict, OrderedDict)):
            self.append(data)
        elif isinstance(data, list):
            self.extend(data)
        else:
            raise TypeError('Unsupported data type {}'.format(data))

    def insert(self, idx, data):
        if isinstance(data, ListDict):
            assert len(data) == 0
            data = data.data[0]
        if isinstance(data, (dict, OrderedDict)):
            self.data.insert(idx, data)
        else:
            raise TypeError('Method insert does support for type {}'.format(type(data)))

    def pop(self, idx):
        return self.data.pop(idx)

    def to_dataframe(self):
        return pd.DataFrame(self.data)

    def to_csv(self, path, index=False, **kwargs):
        df = self.to_dataframe()
        df.to_csv(path, columns=self.header, index=index, **kwargs)

    @classmethod
    def load_csv(cls, path, **kwargs):
        if not os.path.isfile(path):
            raise FileExistsError('{} does not exist.'.format(path))
        df = pd.read_csv(path)
        data = df.to_dict('records')
        return cls(data=data, **kwargs)

if __name__ == "__main__":
    lista = ListDict()
    lista.to_csv(path="wat.csv")
