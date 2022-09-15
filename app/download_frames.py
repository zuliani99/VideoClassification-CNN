import functools
import os
from multiprocessing import Pool
import cv2
import yt_dlp
import time
import tqdm
from termcolor import colored

def process_video(url, skip_frames, skip_start_end_frames, directory, url_id, total_frames, labels):
	cap = cv2.VideoCapture(url)
	x = 0
	count = skip_start_end_frames
 
	while count < total_frames:
		cap.set(1, count)
		ret, frame = cap.read()
  
		if not ret: break

		for lab in labels:
			filename = os.path.join('data', directory, lab, f'{url_id}_{x}.png')
			cv2.imwrite(filename.format(count), frame)
   
		x += 1
		count += skip_frames
  
	cap.release()


def take_shots_from_url(directory, percentage_of_frames, video_url):
	url, labels = video_url
	url_id = url.split('=')[1]
 
	try:
		ydl_options = {"quiet": True, 'verbose': False, 'youtube_include_dash_manifest': False}

		with yt_dlp.YoutubeDL(ydl_options) as ydl:
			info_dict = ydl.extract_info(url, download=False)

			video_length_or = info_dict['duration'] * info_dict['fps']
			frames_out_perc = (video_length_or * 10) // 100
			video_length =  video_length_or - (frames_out_perc * 2)
			skip_rate = video_length // min((video_length * percentage_of_frames) // 100, 10) # max 10 frames per video

			format_id = {f['format_id']: f for f in info_dict.get('formats', None)}
			if '160' in list(format_id.keys()):
				print(colored(f'Obtaining frames of: {url_id}, with labels: {labels}', 'cyan'))
				process_video(format_id['160'].get('url', None), skip_rate, frames_out_perc, directory, url_id, video_length, labels)
    
	except Exception as e: print(e)

		
	

def download_frames(url_lists):
	pool = Pool(os.cpu_count())

	for url_list in url_lists:
		urls, directory = url_list
		print(colored(f'--- Downloading {directory} ---', 'green'))
		start_time = time.time()
		list(tqdm.tqdm(pool.imap(functools.partial(take_shots_from_url, directory, 3), list(urls.items())), total=len(list(urls.items()))))
		print(colored(f'--- Download of {directory} took: {time.time() - start_time} seconds ---\n', 'green'))
	pool.close()
 
 
def get_label_name(LABEL, id): return LABEL[int(id)]


def get_dataset():
	TRAIN, TEST = {}, {}
	DATA = [TRAIN, TEST]
	LABELS = []
	path = './data/sports-1m-dataset-master/original'
 
	with open('./data/sports-1m-dataset-master/labels.txt') as f_labels:
		LABELS = f_labels.read().splitlines()

	for idx, f in enumerate(os.listdir(path)):
		with open(os.path.join(path, f)) as f_txt:
			lines = f_txt.readlines()
			for line in lines:
				splitted_line = line.split(' ')
				label_indices = splitted_line[1].rstrip('\n').split(',') 
				DATA[idx][splitted_line[0]] = list(map(functools.partial(get_label_name, LABELS), label_indices))

	for label in LABELS:
		os.makedirs(os.path.join('data', 'train_shots', label))
		os.makedirs(os.path.join('data', 'test_shots', label))

	return DATA, LABELS
