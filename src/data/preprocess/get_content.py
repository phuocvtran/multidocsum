from glob import glob
import os


os.makedirs('data', exist_ok=True)
for folder in glob('data/raw/ViMs/original/*'):
    folder_name = folder.split('/')[-1]
    os.makedirs(f'data/interim/content/{folder_name}', exist_ok=True)
    for file in glob(f'{folder}/original/*'):
        file_name = file.split('/')[-1]
        with open(file, 'r') as f:
            line_list = f.read().splitlines()
        try:
            content_index = line_list.index('Content:')
        except ValueError:
            for index, item in enumerate(line_list):
                if item.count('Content:') != 0:
                    content_index = index - 1
                    line_list[index] = line_list[index].replace('Content:', '')
                    break

        line_list = line_list[content_index+1:-1]
        with open(f'data/interim/content/{folder_name}/{file_name}', 'w') as f:
            f.write('\n'.join(line_list))