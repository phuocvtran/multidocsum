import numpy as np
from typing import Dict, List, Tuple, Callable
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import Utils


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
    def get_pos_score(doc_list: List[str], pos_list: List[str], C_max: int = 10) -> List[float]:
        df = pandas.DataFrame({'doc': doc_list, 'pos': pos_list})
        group = df.groupby(['doc']).count()

        pos_score = []
        for idx, pos in enumerate(pos_list):
            n = group['pos'][doc_list[idx]]
            pos_score.append((n - pos + 1) / n * C_max)

        return pos_score

    @staticmethod
    def get_first_sent_overlap_score(data_list: List[str], 
                                     pos_list: List[int], 
                                     doc_list: List[str], 
                                     ngram_range: Tuple[int, int] = (1, 1), 
                                     max_features: int = 300, 
                                     func: Callable = np.dot) -> List[float]:
        df = pandas.DataFrame(
            {'data': data_list, 'pos': pos_list, 'doc': doc_list})
        group = df.groupby(['doc'])

        doc_idx = set(doc_list)

        tfidf_dict = {}
        first_sent = {}
        for idx in doc_idx:
            temp_df = group.get_group(idx)
            corpus = temp_df.data.to_list()
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_range, max_features=max_features)
            tfidf_dict[idx] = vectorizer.fit(corpus)

            temp_sent = temp_df.data[temp_df.pos == 1].to_list()[0]
            first_sent[idx] = vectorizer.transform([temp_sent]).toarray()[0]

        overlap_score = []
        for _, row in df.iterrows():
            idx = row.doc
            data = row.data
            fs = first_sent[idx]
            vec = tfidf_dict[idx].transform([data]).toarray()[0]

            osc = func(fs, vec)
            overlap_score.append(osc)

        return overlap_score

    @staticmethod
    def scoring_sentences(raw_data: Dict[str, str],
                          processed_data: Dict[str, str],
                          wc: float = 1.0,
                          wp: float = 1.0,
                          wf: float = 1.0,
                          n_centroid: int = 20,
                          C_max: int = 100,
                          ngram_range: Tuple[int, int] = (1, 1),
                          max_features: int = 300,
                          func: Callable = np.dot) -> pandas.DataFrame:
        data_frame = Utils.get_dataframe(raw_data, processed_data)
        centroids = Utils.get_centroid(data_frame.processed.to_list(
        ), data_frame.doc.to_list(), n_centroid=n_centroid)
        cent_score = Scorer.get_centroid_score(
            data_frame.processed.to_list(), centroids)
        data_frame['centroid_score'] = Utils.scale(cent_score)
        pos_score = Scorer.get_pos_score(
            data_frame.doc.to_list(), data_frame.pos.to_list(), C_max=C_max)
        data_frame['pos_score'] = Utils.scale(pos_score)
        ovl_score = Scorer.get_first_sent_overlap_score(data_frame.processed.to_list(), data_frame.pos.to_list(
        ), data_frame.doc.to_list(), ngram_range=ngram_range, max_features=max_features, func=func)
        data_frame['ovl_score'] = Utils.scale(ovl_score)

        sum_score = []
        for _, row in data_frame.iterrows():
            temp = wc * row.centroid_score + wp * row.pos_score + wf * row.ovl_score
            sum_score.append(temp)

        data_frame['sum_score'] = sum_score

        return data_frame
