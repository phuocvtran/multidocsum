import os
from .data_io import DataIO
from .text_processing import TextProcessor
from ..const import RAW_DATA_PATH, RAW_CLUSTER_POS
from typing import Dict, List, Callable


def download_data() -> None:
    os.system('''
            mkdir -p data/raw
            git clone https://github.com/CLC-HCMUS/ViMs-Dataset.git data/raw
            unzip data/raw/ViMs.zip -d data/raw
            rm data/raw/ViMs.zip
            rm data/raw/README.md
            rm -rf data/raw/.git
    ''')


def _preprocess_n_write(data: Dict[str, List[str]], func: Callable, out_path: str=None, kwargs: Dict[str, str]=None) -> Dict[str, List[str]]:
    res = data
    for key in res:
        temp = []
        for text in data[key]:
            if kwargs is not None:
                temp.append(func(text, **kwargs))
            else:
                temp.append(func(text))
        res[key] = temp
        
    if out_path is not None:
        DataIO.write(res, out_path)
    
    return res


def preprocess() -> None:
    print('Loading data...')
    raw_data = DataIO.load_raw_(RAW_DATA_PATH, RAW_CLUSTER_POS)

    print('Processing data...')
    
    # get content
    contents = _preprocess_n_write(raw_data, TextProcessor.get_content_from_raw, out_path='data/interim/extract_content')

    # raw sentence seg
    _preprocess_n_write(contents, TextProcessor.sent_tokenize, out_path='data/interim/00_RAW_SENT_SEG')

    # word segmentation
    word_seg_data = _preprocess_n_write(contents, TextProcessor.word_tokenize, out_path='data/interim/word_tokenize')

    # lower
    lower_data = _preprocess_n_write(word_seg_data, TextProcessor.lower)

    # remove stop words
    with open('data/external/stopwords.txt', 'r') as f:
        stop_words = f.read().splitlines()
    kwargs = {'stop_words': stop_words}
    no_stopwords_data = _preprocess_n_write(lower_data, TextProcessor.remove_stopwords, out_path='data/interim/remove_stop_word', kwargs=kwargs)
    
    # sentence segmentation
    sent_seg_data = _preprocess_n_write(no_stopwords_data, TextProcessor.sent_tokenize, out_path='data/interim/sentence_seg')
    
    # remove punctuation
    with open('data/external/punctuation.txt', 'r') as f:
        punctuation = f.read()
    kwargs = {'punctuation': punctuation}
    no_punct_data = _preprocess_n_write(sent_seg_data, TextProcessor.remove_punctuation, out_path='data/interim/remove_punctuation', kwargs=kwargs)
    
    DataIO.write_(no_punct_data, 'data/processed')