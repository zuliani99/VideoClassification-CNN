import yt_dlp, os
from tqdm.notebook import tqdm
import cv2, shutil, random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pandas as pd
import numpy as np
import argparse, requests, zipfile, io

import logging
logger = logging.getLogger(__name__)



DATA = {'train_partition.txt': {}, 'test_partition.txt': {}}

train_dict = {}
test_dict = {}

# Choosen labels video
LABELS = {
    '398': 'rugby',
    '300': 'formula racing',
    '389': 'street football',
    '368': 'basketball',
    '338': 'hockey',
    '277': 'motocross',
    '283': 'trial',
    '269': 'motorcycle racing',
    '258': 'horse racing',
    '200': 'bodybuilding',
}


def download_zip_file(url: str, zip_f_name: str) -> None:
    '''
    PURPOSE:
      Download the zip file from the given url and extract it to the specified path

    TAKES:
      - url: the url of the zip file
      - zip_f_name: the name of the zip file

    RETURNS:
      None
    '''

    response = requests.get(url, stream=True)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall(zip_f_name)
    


def extract_frames(capture, directory, idx_bag, start_frame, end_frame):
    '''
    PORPOUSE:
      Extract frames from a given youtube video

    TAKES:
      - capture: cv2.VideoCapture(url) variable, with url being the link to the video
      - directory: describe the saving directory
      - idx_bag: indicate the index of the actual bag of shots
      - start_frame: indicates the starting frame
      - end_frame: indicates the ending frame

    RETURNS:
      True or False depending the presence of errors
    '''

    count = start_frame

    # Set the next frame to download to 'count'
    capture.set(cv2.CAP_PROP_POS_FRAMES, count)
    os.makedirs(f'{directory}/bag_of_shots{str(idx_bag)}') # Create the relative directory

    # Download the frame until we do not reach the end_frame
    while count < end_frame:

        ret, frame = capture.read() # Read the frame

        if not ret or frame is None: # In case there are errors, delete the after mentioned directory of bag of shots
            shutil.rmtree(f'{directory}/bag_of_shots{str(idx_bag)}')
            return False

        # Save the readed frame in the directory of bag of shots resizing it to be 178x178
        filename = f'{directory}/bag_of_shots{str(idx_bag)}/shot{str(count - start_frame)}.png'
        write_res = cv2.imwrite(filename, cv2.resize(frame, (178, 178), interpolation = cv2.INTER_AREA))
        if write_res:
            count += 1    # If there is no error I increment the count
        else:   # Otherwise I delete the all bag of shots returning Falses
            shutil.rmtree(f'{directory}/bag_of_shots{str(idx_bag)}')
            return False

    return True

def video_to_frames(args: argparse.Namespace, dataset_path: str, video_url: str, label_id: str, directory: str, percentage_of_bags: int):
    '''
    PORPOUSE:
      Determine the amount of frame and bag of shots we want to downloads in each video, and perform the download of them

    TAKES:
      - video_url: complete url link to a specific YouTubbe video
      - directory: train or test string
      - label_id: label id
      - idx_bag: indicate the index of the actual bag of shots
      - percentage_of_bags: indicate the percentage of bags of shots that we want to download

    RETURNS: 
      ret_dictionary: A dictionary containing as keys the path to a specific bag of shots and as value the list of labels for the 
                      relative bag, the last element of this list represent the number of frames we have downloaded
    '''

    url_id = video_url.split('=')[1]
    path_until_url_id = f'{dataset_path}/{directory}/{url_id}'
    
    ret_dictionary = {}

    try:   

        # Setting up the dictionary options for yd-dlp
        ydl_opts = {
            'ignoreerrors': True,
            'quiet': True,
            'nowarnings': True,
            'ignorenoformatserror': True,
            'verbose': False,
            'cookies': '/content/all_cookies.txt', # Include all saved cookies to prevent the 'TOO MANY REQUESTS' error
            #https://stackoverflow.com/questions/63329412/how-can-i-solve-this-youtube-dl-429
        }

        ydl = yt_dlp.YoutubeDL(ydl_opts)
        info_dict = ydl.extract_info(video_url, download=False) # Extracting the video infromation
        
        if(info_dict is not None):

            formats = info_dict.get('formats', None)

            format_id = {}
            for f in formats: format_id[f['format_id']] = f

            # I only consider the format_id 160 with indicates the 144p resolution
            if '160' in list(format_id.keys()): 
                
                video = format_id['160'] # Get all the details of the 144p resolution video
                fps = video.get('fps', None)
                url = video.get('url', None)

                # I must have a least 20 frames per seconds since I take half of second bag of shots for every video
                if(fps >= 20 and url != video.get('manifest_url', None)):
                    
                    capture = cv2.VideoCapture(url) # Initialize VideoCapture variable
                    video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) # Get the total frame count
                    shots = fps // 2

                    # Excluding the initial and final choosen percentage of each video to avoid noise
                    to_ignore = (video_length * args.percentage_to_ignore) // 100
                    new_len = video_length - (to_ignore * 2)
                    tot_stored_bags = ((new_len // shots) * percentage_of_bags) // 100   # ((total_possbile_bags // shots) * percentage_of_bags) // 100

                    if tot_stored_bags == 0: tot_stored_bags = 1 # I take at least a bag of shots

                    # Computing the skip rate between bags
                    skip_rate_between_bags = (new_len - (tot_stored_bags * shots)) // (tot_stored_bags + 1)

                    chunks = [[to_ignore + ((bag * skip_rate_between_bags) + (shots * (bag - 1))),
                              to_ignore + (bag * (skip_rate_between_bags + shots))] for bag in range(1, tot_stored_bags + 1)]
                    # Sequence of [[start_frame, end_frame], [start_frame, end_frame], [start_frame, end_frame], ...]

                    if not os.path.exists(path_until_url_id): os.makedirs(path_until_url_id)
                    # Create the the folder that will contain all the bag of shots

                    capture = cv2.VideoCapture(url) # Initialize VideoCapture variable
                    valid_chunks = 0
 
                    for idx_bag, f in enumerate(chunks): # For each chunks

                        # In case the download of the bag of shots succedeed
                        if(extract_frames(capture, path_until_url_id, idx_bag, f[0], f[1])):
                              
                              l = np.array([label_id, shots], dtype=object)
                              valid_chunks += 1

                              ret_dictionary[f'{directory}/{url_id}/bag_of_shots{str(idx_bag)}'] = l.tolist() # Populate the new dictionary row

                    # In case we do not have downloaded any chunks delete the directory and all its content
                    if valid_chunks == 0: shutil.rmtree(path_until_url_id)

                    capture.release()

        return ret_dictionary


    except Exception as e:
        # If an exception rised delete the directory with all its content and return ret_dictionary
        if os.path.exists(path_until_url_id): shutil.rmtree(path_until_url_id)
        return ret_dictionary


def get_inital_path(args: argparse.Namespace) -> str:
    '''
    PURPOSE:
      Get the initial path of the dataset

    RETURNS:
      The initial path of the dataset
    '''

    initial_ds_path = os.path.join(os.path.dirname(__file__), 'dataset')
    if not os.path.exists(initial_ds_path): os.makedirs(initial_ds_path)
    our_dataset_folder = os.path.join(initial_ds_path, f'{args.percentage_train_test}_{args.percentage_bag_shots}_{args.percentage_to_ignore}')
    if not os.path.exists(our_dataset_folder): os.makedirs(our_dataset_folder)
    return initial_ds_path, our_dataset_folder


def get_dataset(args: argparse.Namespace) -> None:
    initial_ds_path, our_dataset_folder = get_inital_path(args)
    

    path = os.path.join(initial_ds_path, 'sports-1m-dataset-master/original')
    if not os.path.exists(path):
        download_zip_file(
            url='https://github.com/gtoderici/sports-1m-dataset/archive/refs/heads/master.zip',
            zip_f_name=os.path.join(initial_ds_path, 'sports-1m-dataset-master.zip')
        )
    else: shutil.rmtree(path)  # Remove the existing directory if it exists
    

    
    # Populate the DATA dictionary by reading the train and test files
    for f in os.listdir(path):
        with open(path + '/' + f) as f_txt:
            lines = f_txt.readlines()
            for line in lines:
                splitted_line = line.split(' ')
                label_indices = splitted_line[1].rstrip('\n').split(',') 
                
                if(len(label_indices) == 1): # Skip in case the video has more than one label
                    label = label_indices[0]
                    if label in list(LABELS.keys()): # Skip in case the label is not one we want to consider
                        DATA[f][splitted_line[0]] = [int(label)] #list(map(int, label_indices))


    TRAIN = DATA['train_partition.txt']
    TEST = DATA['test_partition.txt']
    logger.info('Original Train Test length: %d, %d', len(TRAIN), len(TEST))

    # Sample a subset of percentage_train_test
    TRAIN = dict(random.sample(list(TRAIN.items()), (len(TRAIN) * args.percentage_train_test) // 100))
    TEST = dict(random.sample(list(TEST.items()), (len(TEST) * args.percentage_train_test) // 100))

    logger.info(f'Sampling {args.percentage_train_test} % of Train & Test datasets, updated length: %d, %d', len(TRAIN), len(TEST))        

    for ds_type, lab in [(TRAIN, 'Train'), (TEST, 'Test')]:
        dataset_path = os.path.join(our_dataset_folder, lab.lower())
        if not os.path.exists(dataset_path): os.makedirs(dataset_path)

        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
            with tqdm(total = len(ds_type.items()), leave=False, desc=f'Downloading {lab} Dataset') as progress: # Setting up the progress bar

                futures = [] # Array of features

                for url, label_id in ds_type.items():
                    future = pool.submit(video_to_frames, args, dataset_path, url, label_id, lab, args.percentage_bag_shots) # Assign the download
                    future.add_done_callback(lambda p: progress.update()) # Update the progrress bar
                    futures.append(future) # Append the feature to the features array

                for future in futures:
                    if len(future.result()) > 0: # In case the result of the featue is not empty
                        train_dict.update(future.result()) # Append the result to the final dictionaty
                        count += 1
        


    # Save train_dict and test_dict as parquet files using a for loop
    for split_name, data_dict in [('train', train_dict), ('test', test_dict)]:
        df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=int).reset_index(level=0)
        col_map = {idx: list(LABELS.keys())[idx] for idx in df.columns[1:-1]}
        col_map[10] = 'shots'
        df = df.rename(columns=col_map)
        df.columns = df.columns.astype(str)
        df.to_parquet(f'{dataset_path}/{split_name}.parquet', index=True)

    with open(f'{dataset_path}/labels.csv', 'w') as f: # Writing a csv file for the labels
        for key in LABELS.keys():
            f.write("%s,%s\n" % (key, LABELS[key]))



def check_consistency(args: argparse.Namespace) -> tuple:
    train_dict = {}
    test_dict = {}
    labels = {}

    _, our_dataset_folder = get_inital_path(args)

    train_df = pd.read_parquet(os.path.join(our_dataset_folder, 'train', 'train.parquet')) # Read the parquet train dataset
    test_df = pd.read_parquet(os.path.join(our_dataset_folder, 'test', 'test.parquet')) # Read the parquet test dataset


    for string, df, dic in zip(('train', 'test'), (train_df, test_df), (train_dict, test_dict)):
        for k, v in tqdm(df.T.items(), total=len(list(df.T.items())), desc=f'Populating {string} dictionary'):
            values = v.to_numpy()
            dic[values[0]] = list(values[1:]) # Populate the dictionary

    with open(f'{our_dataset_folder}/labels.csv', 'r') as label_csv: # Read the labels and create a dictionary
        lines = label_csv.readlines()
        for line in tqdm(lines, total=len(lines), desc=f'Populating the labels dictionary'):
            splitted_line = line.split(',')
            labels[splitted_line[0]] = splitted_line[1].strip()


    logger.info(f'\nSampled {args.percentage_train_test} % of Train & Test datasets')
    logger.info(f'Number of train bags of shots: {len(train_df)}')
    logger.info(f'Number of test bags of shots: {len(test_dict)}')
    logger.info(f'Number of labels: {len(labels)}')
    
    return train_dict, test_dict, labels, train_df, test_df