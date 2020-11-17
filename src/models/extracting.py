import pandas
from ..features.utils import Utils


class Extractor:
    @staticmethod
    def remove_redundancy(data_frame: pandas.DataFrame, compression_rate: float=0.05) -> pandas.DataFrame:
        df = data_frame.sort_values(['sum_score'], ascending=False).reset_index(drop=True)
        wr = df.sum_score.to_list()[0]
        
        n_extract = round(df.raw.shape[0] * compression_rate)
        
        
        while True:
            pre_extract = df.raw.copy().loc[:n_extract]
            for index, curr_row in df.iterrows():
                if index == 0:
                    continue
                for midx, pre_row in df.iterrows():
                    if midx == index:
                        break
                    ovw = Utils.count_overlapping_word(curr_row.processed, pre_row.processed)
                    len_curr = len(curr_row.processed)
                    len_pre = len(pre_row.processed)
                    
                    df.sum_score.iloc[index] -= wr * (2 * ovw / (len_curr + len_pre))
                
            df = df.sort_values(['sum_score'], ascending=False).reset_index(drop=True)
            curr_extract = df.raw.loc[:n_extract]
            if pre_extract.equals(curr_extract):
                break
            
        return df