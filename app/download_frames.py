import functools
import os
from multiprocessing import Pool
import cv2
import youtube_dl
import shutil
import time
from termcolor import colored


#change package to remove the verbose mode


def process_video(url, skip_frames, directory, id, total_frames):
    cap = cv2.VideoCapture(url)
    x = 0
    count = 0
    while count < total_frames:
        ret, frame = cap.read()
        
        if not ret: 
            print('Error can not open the video')
            break
        
        #filename = f'{directory}/{id}/shot{str(x)}.png'
        filename = os.path.join('/data', directory, id, f'shot{str(x)}.png')
        x += 1
        cv2.imwrite(filename.format(count), cv2.resize(frame, (256, 144), interpolation = cv2.INTER_AREA))
        count += skip_frames
    cap.release()



def take_shots_from_url(directory, percentage_of_frames, length, ids_video_url):
    count, video_url = ids_video_url
    id = video_url.split('=')[1]
    try:
        if not os.path.exists(os.path.join('data',directory, id)):
            os.makedirs(os.path.join('data',directory, id))

        ydl = youtube_dl.YoutubeDL({"quiet": True})
        info_dict = ydl.extract_info(video_url, download=False)

        video_length = info_dict['duration'] * info_dict['fps']
        tot_stored_frames = min((video_length * percentage_of_frames) // 100, 10)
        skip_rate = video_length // tot_stored_frames

        resolution_id = ['160', '133', '134', '135', '136']
        format_id = {f['format_id']: f for f in info_dict.get('formats', None)}
        for res in resolution_id:
            if res in list(format_id.keys()):
                video = format_id[res]
                url = video.get('url', None)
                if(video.get('url', None) != video.get('manifest_url', None)):
                    print(colored((f'{count} / {length}'), 'green') + f' -> Obtaining frames of {video_url}, length {info_dict["duration"]}, fps: {info_dict["fps"]}, resolution: {res}p, skip rate: {skip_rate}')
                    process_video(url, skip_rate, directory, id, info_dict['duration'] * info_dict['fps'])
                    break
            else:
                print(f'No {res}p resolution found, trying a higher one')
    except Exception as e:
        print(e)
        shutil.rmtree(os.path.join('data', directory, id))


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
        start_time = time.time()
        pool.map(functools.partial(take_shots_from_url, directory, 3, len(urls)), enumerate(urls))
        pool.close()
        print(f"--- {time.time() - start_time} seconds ---")
