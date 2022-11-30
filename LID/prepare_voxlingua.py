import os
import json
import shutil
import random
import logging
import glob
from speechbrain.utils.data_utils import get_all_files
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
SAMPLERATE = 16000

def create_json(wav_list, json_file):
    	# Processing all the wav files in the list
    	json_dict = {}
    	for wav_file in wav_list:

        	# Reading the signal (to retrieve duration in seconds)
        	signal = read_audio(wav_file)
        	duration = signal.shape[0] / SAMPLERATE

        	# Manipulate path to get relative path and uttid
        	path_parts = wav_file.split(os.path.sep)
        	uttid, _ = os.path.splitext(path_parts[-1])
        	lang_id = os.path.splitext(path_parts[-2])[0]
        	relative_path = os.path.join("{data_root}", *path_parts[-4:])

        	# Create entry for this utterance
        	json_dict[uttid] = {
            		"wav": relative_path,
            		"length": duration,
            		"lang_id": lang_id,
        	}

    	# Writing the dictionary to the json file
    	with open(json_file, mode="w") as json_f:
        	json.dump(json_dict, json_f, indent=2)

    	logger.info(f"{json_file} successfully created!")

def split_sets(wav_list, split_ratio):
    	# Random shuffle of the list
    	random.shuffle(wav_list)
    	tot_split = sum(split_ratio)
    	tot_snts = len(wav_list)
    	data_split = {}
    	splits = ["train", "valid"]

    	for i, split in enumerate(splits):
        	n_snts = int(tot_snts * split_ratio[i] / tot_split)
        	data_split[split] = wav_list[0:n_snts]
        	del wav_list[0:n_snts]
        
    	data_split["test"] = wav_list

    	return data_split

def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True

def prepare_voxlingua(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    split_ratio=[80, 10, 10],
):
    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return
	
    train_folder = os.path.join(data_folder, "voxLingua", "new_train")
	
    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    
    extension = [".wav"]
    wav_list = get_all_files(train_folder, match_and=extension)
    	
    # Random split the signal list into train, valid, and test sets.
    data_split = split_sets(wav_list, split_ratio)
    	
    # Creating json files
    create_json(data_split["train"], save_json_train)
    create_json(data_split["valid"], save_json_valid)
    create_json(data_split["test"], save_json_test)
