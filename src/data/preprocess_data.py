import glob
import os
from underthesea import word_tokenize
import numpy as np
import string
import pandas as pd
from pyvi import ViTokenizer
import time

# path_name: list đường dẫn tới các cụm văn bản khác nhau.
path_name = np.sort(glob.glob('data/interim/sentence-segments/*'))

# path_file_name: list danh sách đường dẫn tới file txt.
path_file_name = [np.sort(glob.glob(path+'/*.txt')) for path in path_name ]

def load_file(path):
    """
    path: đường dẫn tới nơi chứa file txt
    text: list danh sách chưa các câu được đọc từ văn bản
    """
    text = []
    with open(path, 'r') as f:
         text = f.readlines()
    text = [word.replace('\n', '') for word in text]
    return text

def word_token(text):
    """
    text: một mảng chứa các câu của một văn bản
    word_token: là danh sách các list từ của từng câu trong văn bản
    """ 
    word_token= []
    for sent in text:
        word_token.append(word_tokenize(sent, format="text"))
    return word_token

def remove_punctuation(text):
    text_ = []
    for sent in text:
        sent_ = [word.lower() for word in sent if word not in string.punctuation or word == '_' or word == '/']
        sent_ = ''.join(sent_)
        text_.append(sent_)
    return text_

def write_txt(path_name, cluster_name, file_name, text):
    """
    path_name: đường dẫn tới thư mục chứa cluster
    cluster_name: tên của nhóm văn bản
    file_name: tên file
    text: list chứa nội dung từng câu của văn bản đã được xử lý
    
    output: là 1 file txt gồm:
         idx: là vị trí của câu
         sentence: nôị dung câu sau khi xử lý

    """
    with open(path_name+f'/{cluster_name}/{file_name}', 'w') as f:
        f.write("idx, sentence\n")
        for i in range(len(text)):
            f.write(str(i)+','+text[i] + '\n')

def stop_word(text, stop_word):
    text_ = []
    for sent in text:
        sent_ = [word for word in sent.split() if word not in stop_word]
        sent_ = ' '.join(sent_)
        text_.append(sent_)
    return text_

    


if __name__ == "__main__":    
    stop_word_ = load_file('data/external/stop_word.txt')
    ck = 'Cluster_001'
    for cluster_path in path_file_name:
        for file_path in cluster_path:
            if ck != os.path.basename(os.path.dirname(file_path)):
                print(ck)
                ck = os.path.basename(os.path.dirname(file_path))
            text = remove_punctuation(word_token(load_file(file_path))) # load file và tiền xử lý
            text = stop_word(text, stop_word_)
            cluster_name = os.path.basename(os.path.dirname(file_path)) # lấy tên cluster
            file_name = os.path.basename(file_path)                     # lấy tên file
            path_name = 'data/preprocess'                               # tạo đường dẫn lưu file
            write_txt(path_name, cluster_name, file_name, text)         # lưu file dạng txt
