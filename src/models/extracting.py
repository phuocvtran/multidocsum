import pandas
from ..features.utils import Utils
from ..data.data_io import DataIO
from ..features.scoring import Scorer
from typing import List, Tuple, Callable
import numpy as np


class MEAD:
    def __init__(self,
                stop_words_path: str,
                punc_path: str,
                remove_redundant_sent: bool = True,
                wc: float=1.0,
                wp: float=1.0,
                wf: float=1.0,
                n_centroid: int=20,
                C_max: int=100,
                ngram_range: Tuple[int, int]=(1, 1),
                max_features: int=300,
                func: Callable=np.dot) -> None:
        self.stop_words_path = stop_words_path
        self.punc_path = punc_path
        self.remove_redundant_sent = remove_redundant_sent
        self.wc = wc
        self.wp = wp
        self.wf = wf
        self.n_centroid = n_centroid
        self.C_max = C_max
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.func = func
        
    def remove_redundancy(self, data_frame: pandas.DataFrame, n_extract: int) -> pandas.Series:
        df = data_frame.sort_values(
            ['sum_score'], ascending=False).reset_index(drop=True)
        wr = df.sum_score[0]

        while True:
            pre_extract = df.raw.copy().loc[:n_extract].sort_values().reset_index(drop=True)
            df['temp_score'] = df.sum_score.to_list()
            for index, curr_row in df.iterrows():
                if index == 0:
                    continue
                for midx, pre_row in df.iterrows():
                    if midx == index:
                        break
                    ovw = Utils.count_overlapping_word(
                        curr_row.processed, pre_row.processed)
                    len_curr = len(curr_row.processed)
                    len_pre = len(pre_row.processed)

                    df.temp_score.iloc[index] -= wr * \
                        (2 * ovw / (len_curr + len_pre))

            df = df.sort_values(
                ['temp_score'], ascending=False).reset_index(drop=True)
            curr_extract = df.raw.loc[:n_extract].sort_values().reset_index(drop=True)
            if pre_extract.equals(curr_extract):
                break

        return df.raw.loc[:n_extract]

    def extract(self, 
                data_path: str,
                compression_rate: float = 0.05,
                save_as: str = None) -> List[str]:
        data = DataIO.load(data_path)
        processed_data, data = Utils.preprocess(data,
                                                stop_words_path=self.stop_words_path,
                                                punc_path=self.punc_path)
        df = Scorer.scoring_sentences(data, processed_data, wc=self.wc, wp=self.wp, wf=self.wf, n_centroid=self.n_centroid,
                                      C_max=self.C_max, ngram_range=self.ngram_range, max_features=self.max_features, func=self.func)

        n_extract = round(df.raw.shape[0] * compression_rate)

        if self.remove_redundant_sent:
            res_series = self.remove_redundancy(
                df, n_extract=n_extract)
        else:
            res_series = df.sort_values(
                ['sum_score'], ascending=False).reset_index().raw.loc[:n_extract]

        if save_as is not None:
            DataIO.write(res_series.to_list(), save_as)

        return res_series.to_list()
