import os
from underthesea import sent_tokenize
from glob import glob


os.makedirs('data/interim/sentence-segments', exist_ok=True)
for folder in glob('data/interim/content/*'):
    folder_name = folder.split('/')[-1]
    os.makedirs(f'data/interim/sentence-segments/{folder_name}', exist_ok=True)
    for f in glob(f'data/interim/content/{folder_name}/*'):
        file_name = f.split('/')[-1]
        with open(f, 'r') as txt_file:
            text = txt_file.read().replace('\n', ' ')
        sents = sent_tokenize(text)
        with open(f'data/interim/sentence-segments/{folder_name}/{file_name}', 'w') as ofile:
            ofile.write('\n'.join(sents))