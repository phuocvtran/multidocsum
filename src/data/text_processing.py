import underthesea
from pyvi import ViTokenizer
from typing import List


class TextProcessor:
    @staticmethod
    def lower(text: str) -> str:
        return text.lower()
    
    @staticmethod
    def get_content_from_raw(text: str,) -> str:
        lines = text.split('\n')
        
        try:
            content_index = lines.index('Content:')
        except ValueError:
            for index, item in enumerate(lines):
                if item.count('Content:') != 0:
                    content_index = index - 1
                    lines[index] = lines[index].replace('Content:', '')
                    break
                
        return ' '.join(lines[content_index+1:-1])

    @staticmethod
    def sent_tokenize(text: str) -> str:
        return ' \n '.join(underthesea.sent_tokenize(text))

    @staticmethod
    def word_tokenize(text: str) -> str:
        temp = []
        for sent in text.split(' \n '):
            temp.append(ViTokenizer.tokenize(sent))
        return ' \n '.join(temp)

    @staticmethod
    def remove_stopwords(text: str, stop_words: List[str]) -> str:
        res = []
        for sent in text.split(' \n '):
            temp = []
            for word in sent.split():
                if word not in stop_words:
                    temp.append(word)
            res.append(' '.join(temp))
                
                
        return ' \n '.join(res)
    
    @staticmethod
    def remove_punctuation(text: str, punctuation: str) -> str:
        res = []
        for sent in text.split(' \n '):
            temp = []
            for word in sent.split():
                if word not in punctuation:
                    temp.append(word)
            res.append(' '.join(temp))
                
                
        return ' \n '.join(res)