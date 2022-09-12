import functools
import os
from multiprocessing import Pool
import cv2
import yt_dlp
import shutil
import time
import tqdm
from termcolor import colored


def process_video(url, skip_frames, directory, url_id, total_frames, lab):
    cap = cv2.VideoCapture(url)
    x = 0
    count = 0
    while count < total_frames:
        ret, frame = cap.read()
        
        if not ret: 
            shutil.rmtree(os.path.join('data',directory, str(lab)))
            break
        
        filename = os.path.join('data', directory, str(lab), f'{url_id}_{str(x)}.png')
        cv2.imwrite(filename, cv2.resize(frame, (256, 144), interpolation = cv2.INTER_AREA))
        
        x += 1
        count += skip_frames
  
    cap.release()



def take_shots_from_url(directory, percentage_of_frames, video_url): 
        url, labels = video_url
        url_id = url.split('=')[1]
        
        for lab in labels:
            if not os.path.exists(os.path.join('data', directory, str(lab))):
                os.makedirs(os.path.join('data',directory, str(lab)))
        
        try:
            ydl_options = {"quiet": True, 'verbose': False}

            with yt_dlp.YoutubeDL(ydl_options) as ydl:
                info_dict = ydl.extract_info(url, download=False)

                video_length = info_dict['duration'] * info_dict['fps']
                tot_stored_frames = min((video_length * percentage_of_frames) // 100, 10)
                skip_rate = video_length // tot_stored_frames

                resolution_id = ['160', '133', '134', '135', '136']
                format_id = {f['format_id']: f for f in info_dict.get('formats', None)}
                for res in resolution_id:
                    if res in list(format_id.keys()):
                        video = format_id[res]
                        url_dict = video.get('url', None)
                        if(video.get('url', None) != video.get('manifest_url', None)):
                            for lab in labels: process_video(url_dict, skip_rate, directory, url_id, video_length, lab)
                            break
        except Exception as e:
            for lab in labels: shutil.rmtree(os.path.join('data',directory, str(lab)))


def get_dataset():
    TRAIN, TEST = {}, {}
    DATA = [TRAIN, TEST]
    LABELS = []
    path = './data/sports-1m-dataset-master/original'

    for idx, f in enumerate(os.listdir(path)):
        with open(os.path.join(path, f)) as f_txt:
            lines = f_txt.readlines()
            for line in lines:
                splitted_line = line.split(' ')
                label_indices = splitted_line[1].rstrip('\n').split(',') 
                DATA[idx][splitted_line[0]] = list(map(int, label_indices))

    with open('./data/sports-1m-dataset-master/labels.txt') as f_labels:
        LABELS = f_labels.read().splitlines()

    return DATA, LABELS
    


def download_frames(url_lists):
    pool = Pool(os.cpu_count())

    for url_list in url_lists:
        urls, directory = url_list
        print(colored(f'--- Downloading {directory} ---', 'green'))
        start_time = time.time()
        list(tqdm.tqdm(pool.imap(functools.partial(take_shots_from_url, directory, 3), list(urls.items())), total=len(list(urls.items()))))
        print(colored(f'--- Download of {directory} took: {time.time() - start_time} seconds ---\n', 'green'))
    
    pool.close()
