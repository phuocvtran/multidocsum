from glob import glob
import os
from typing import Dict, List


class DataIO:
    @staticmethod
    def load(path: str, key_pos: int=None) -> Dict[str, List[str]]:
        data = {}
        for fi in glob(path):
            if key_pos is None:
                cluster_key = '/'.join(fi.split('/')[:-1])
            else:    
                cluster_key = fi.split('/')[key_pos]

            if cluster_key not in data:
                data[cluster_key] = []

            with open(fi) as f:
                text = f.read()
            data[cluster_key].append(text)

        return data
    
    @staticmethod
    def write(data: Dict[str, List[str]], path: str) -> None:
        os.makedirs(path, exist_ok=True)
        for key in data:
            cluster_path = os.path.join(path, key)
            os.makedirs(cluster_path, exist_ok=True)
            for index, text in enumerate(data[key]):
                file_path = os.path.join(cluster_path, f'{index}.txt')
                with open(file_path, 'w') as f:
                    f.write(text)

            