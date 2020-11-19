import pandas
from ..features.utils import Utils
from ..data.data_io import DataIO
from ..features.scoring import Scorer
from typing import List, Tuple, Callable
import numpy as np


class Extractor:
    @staticmethod
    def remove_redundancy(data_frame: pandas.DataFrame, n_extract: int) -> pandas.DataFrame:
        df = data_frame.sort_values(
            ['sum_score'], ascending=False).reset_index(drop=True)
        wr = df.sum_score.to_list()[0]

        while True:
            pre_extract = df.raw.copy().loc[:n_extract].sort_values()
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
            curr_extract = df.raw.loc[:n_extract].sort_values()
            if pre_extract.equals(curr_extract):
                break

        return df.raw.loc[:n_extract]

    @staticmethod
    def extract(data_path: str,
                stop_words_path: str,
                punc_path: str,
                remove_redundacy: bool = True,
                compression_rate: float = 0.05,
                save_as: str = None,
                wc: float = 1.0,
                wp: float = 1.0,
                wf: float = 1.0,
                n_centroid: int = 20,
                C_max: int = 100,
                ngram_range: Tuple[int, int] = (1, 1),
                max_features: int = 300,
                func: Callable = np.dot) -> List[str]:
        data = DataIO.load(data_path)
        processed_data, data = Utils.preprocess(data,
                                                stop_words_path=stop_words_path,
                                                punc_path=punc_path)
        df = Scorer.scoring_sentences(data, processed_data, wc=wc, wp=wp, wf=wf, n_centroid=n_centroid,
                                      C_max=C_max, ngram_range=ngram_range, max_features=max_features, func=func)

        n_extract = round(df.raw.shape[0] * compression_rate)

        if remove_redundacy:
            res_series = Extractor.remove_redundancy(
                df, n_extract=n_extract)
        else:
            res_series = df.sort_values(
                ['sum_score'], ascending=False).reset_index().raw.loc[:n_extract]

        if save_as is not None:
            DataIO.write(res_series.to_list(), save_as)

        return res_series.to_list()
