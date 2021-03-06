import os
from .data_io import DataIO
from .text_processing import TextProcessor
from ..const import RAW_DATA_PATH, RAW_CLUSTER_POS, INTERIM_PATH, PROCESSED_PATH
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


def preprocess_n_write_(data: Dict[str, List[str]], func: Callable, out_path: str=None, kwargs: Dict[str, str]=None) -> Dict[str, List[str]]:
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
        DataIO.write_(res, out_path)
    
    return res


def preprocess(raw_data_path: str=RAW_DATA_PATH, cluster_pos_in_path: int=RAW_CLUSTER_POS, interim_path: str=INTERIM_PATH, processed_path: str=PROCESSED_PATH) -> None:
    print('Loading data...')
    raw_data = DataIO.load_raw_(raw_data_path, cluster_pos_in_path)

    print('Processing data...')
    
    # get content
    contents = preprocess_n_write_(raw_data, TextProcessor.get_content_from_raw, out_path=os.path.join(interim_path ,'extract_content'))

    # sentence segmentation
    sent_seg_data = preprocess_n_write_(contents, TextProcessor.sent_tokenize, out_path=os.path.join(interim_path ,'sentence_seg'))

    # word segmentation
    word_seg_data = preprocess_n_write_(sent_seg_data, TextProcessor.word_tokenize, out_path=os.path.join(interim_path ,'word_tokenize'))

    # lower
    lower_data = preprocess_n_write_(word_seg_data, TextProcessor.lower)

    # remove stop words
    with open('data/external/stopwords.txt', 'r') as f:
        stop_words = f.read().splitlines()
    kwargs = {'stop_words': stop_words}
    no_stopwords_data = preprocess_n_write_(lower_data, TextProcessor.remove_stopwords, out_path=os.path.join(interim_path ,'remove_stop_word'), kwargs=kwargs)
    
    # remove punctuation
    with open('data/external/punctuation.txt', 'r') as f:
        punctuation = f.read()
    kwargs = {'punctuation': punctuation}
    no_punct_data = preprocess_n_write_(no_stopwords_data, TextProcessor.remove_punctuation, out_path=os.path.join(interim_path ,'remove_punctuation'), kwargs=kwargs)
    
    DataIO.write_(no_punct_data, processed_path)