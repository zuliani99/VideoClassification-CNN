#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from download_frames import get_dataset, download_frames

def main():
    if(not os.path.exists('./data/sports-1m-dataset-master')):
        # DOWNOAD ZIP FILES
        os.system("wget --no-verbose https://github.com/gtoderici/sports-1m-dataset/archive/refs/heads/master.zip")

        # EXTRACT AND DELETE THEM
        os.system("unzip -qq -o master.zip -d ./data")
        os.system("rm master.zip")
    
    DATA, LABELS = get_dataset()

    train_url_list, test_url_list = DATA
    
    c = download_frames([(train_url_list, 'train_shots'), (test_url_list, 'test_shots')])
    print(str(c))
    
    
def blame_directory(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if (os.path.isfile(file_path) or os.path.islink(file_path)) and not file_path.endswith('.gitignore'):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Key Interrupted')
        for dir in (os.path.join('data', 'train_shots'), os.path.join('data', 'test_shots')): blame_directory(dir)