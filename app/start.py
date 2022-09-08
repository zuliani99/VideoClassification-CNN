#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from download_frames import get_dataset, download_frames

if __name__ == "__main__":
	# DOWNOAD ZIP FILES
	#os.system("wget --no-verbose https://github.com/gtoderici/sports-1m-dataset/archive/refs/heads/master.zip")

	# EXTRACT AND DELETE THEM
	#os.system("unzip -qq -o master.zip -d ./data")
	#os.system("rm master.zip")
	
	DATA, LABELS = get_dataset()

	train_url_list = list(DATA[0].keys())
	test_url_list = list(DATA[1].keys())
	
	download_frames([(train_url_list, 'train_shots'), (test_url_list, 'test_shots')])