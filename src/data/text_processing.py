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
        return ViTokenizer.tokenize(text)

    @staticmethod
    def remove_stopwords(text: str, stop_words: List[str]) -> str:
        return ' '.join([word for word in text.split() if word not in stop_words])
    
    @staticmethod
    def remove_punctuation(text: str, punctuation: str) -> str:
        res = []
        for word in text.split(' '):
            if word not in punctuation:
                res.append(word)
                
        return ' '.join(res)