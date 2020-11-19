from typing import List, Dict
from math import log


class Tfidf:
    @staticmethod
    def get_tf(data: List[str], doc_idx: List[str]) -> Dict[str, int]:
        n_docs = len(set(doc_idx))
        corpus = ' '.join(data)
        tf = {}
        for word in corpus.split():
            if word not in tf:
                tf[word] = 1
            else:
                tf[word] += 1

        return {word: count / n_docs for (word, count) in tf.items()}

    @staticmethod
    def get_idf(data: List[str], word_list: List[str]) -> Dict[str, float]:
        idf = {}
        N = len(data)
        for word in word_list:
            idf[word] = log(N / Tfidf.get_df(word, data)) + 1

        return idf

    @staticmethod
    def get_tfidf(data: List[str], doc_idx: List[str]) -> Dict[str, float]:
        tf = Tfidf.get_tf(data, doc_idx)
        word_list = [key for key in tf]
        idf = Tfidf.get_idf(data, word_list)

        tfidf = {}
        for word in word_list:
            tfidf[word] = tf[word] * idf[word]

        return tfidf

    @staticmethod
    def get_df(word: str, data: List[str]) -> int:
        df = 0
        for sent in data:
            if word in sent:
                df += 1

        return df
