from typing import Callable, Dict, List, Tuple
import pandas
from .tfidf import Tfidf
from ..data.text_processing import TextProcessor


class Utils:
    @staticmethod
    def do_sth_(data: Dict[str, str], func: Callable, kwargs: Dict[str, str]=None) -> Dict[str, str]:
        res = data
        for key in res:
            text = data[key]
            if kwargs is not None:
                text = func(text, **kwargs)
            else:
                text = func(text)
            res[key] = text
            
        return res
    
    @staticmethod
    def preprocess(data: Dict[str, str], stop_words_path=None, punc_path=None) -> Dict[str, str]:
        contents = Utils.do_sth_(data, TextProcessor.get_content_from_raw)

        # sentence segmentation
        sent_seg_data = Utils.do_sth_(contents, TextProcessor.sent_tokenize)

        raw_sent_seg = sent_seg_data.copy()

        # word segmentation
        word_seg_data = Utils.do_sth_(sent_seg_data, TextProcessor.word_tokenize)

        # lower
        lower_data = Utils.do_sth_(word_seg_data, TextProcessor.lower)

        # remove stop words
        with open(stop_words_path, 'r') as f:
            stop_words = f.read().splitlines()
        kwargs = {'stop_words': stop_words}
        no_stopwords_data = Utils.do_sth_(lower_data, TextProcessor.remove_stopwords, kwargs=kwargs)

        # remove punctuation
        with open(punc_path, 'r') as f:
            punctuation = f.read()
        kwargs = {'punctuation': punctuation}
        no_punct_data = Utils.do_sth_(no_stopwords_data, TextProcessor.remove_punctuation, kwargs=kwargs)

        processed_data = no_punct_data
        
        return processed_data, raw_sent_seg
    
    @staticmethod
    def get_dataframe(raw_data: Dict[str, str], processed_data: Dict[str, str]) -> pandas.DataFrame:
        raw_sent = []
        processed_sent = []
        document = []
        pos =[]

        for key in raw_data:
            raw = raw_data[key].split(' \n ')
            proc = processed_data[key].split(' \n ')
            doc = [key] * len(raw)
            
            raw_sent += raw
            processed_sent += proc
            document += doc
            pos += [i + 1 for i, _ in enumerate(raw)]
            
        return pandas.DataFrame({'raw': raw_sent, 'processed': processed_sent, 'doc': document, 'pos': pos})
    
    @staticmethod
    def get_centroid(data: List[str], n_centroid: int=20) -> List[Tuple[str, float]]:
        tfidf = Tfidf.get_tfidf(data)
        sorted_key = sorted(tfidf, key=tfidf.get, reverse=True)
        sorted_dict = {}
        for key in sorted_key:
            sorted_dict[key] = tfidf[key]
            
        centroids = list(sorted_dict.items())[:n_centroid]
        
        return centroids