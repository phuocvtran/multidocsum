from typing import List, Tuple
import pandas


class Scorer:
    @staticmethod
    def get_centroid_score(data: List[str], centroid_list: List[Tuple[str, float]]) -> List[float]:
        centroid_dict = {key: value for (key, value) in centroid_list}
        
        centroid_score = []
        for sent in data:
            centroid_score.append(0)
            for word in sent.split():
                if word in centroid_dict:
                    centroid_score[-1] += centroid_dict[word]
        
        return centroid_score
    
    @staticmethod
    def get_pos_score(doc_list: List[str], pos_list: List[str], C_max: int=10) -> List[float]:
        df = pandas.DataFrame({'doc': doc_list, 'pos': pos_list})
        group = df.groupby(['doc']).count()
        
        pos_score = []
        for idx, pos in enumerate(pos_list):
            n = group['pos'][doc_list[idx]]
            pos_score.append((n - pos + 1) / n * C_max)
            
        return pos_score