from glob import glob
import os


PATH = 'data/raw/ViMs/original/*/original/*'
def load_data(path: str, key_pos=None: int) -> dict[str, list[str]]:
    dic = {}
    for fi in glob(path):
        if key_pos is None:
            dic_key = '/'.join(fi.split('/')[:-1])
        else:    
            dic_key = fi.split('/')[key_pos]

        if dic_key not in dic:
            dic[dic_key] = []

        with open(fi) as f:
            text = f.read()
        dic[dic_key].append(text)

    return dic