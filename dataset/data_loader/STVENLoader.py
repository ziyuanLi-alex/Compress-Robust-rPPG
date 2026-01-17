"""The dataloader for STVEN pre-training.

It handles pairs of uncompressed (CRF 0) and compressed (CRF > 0) videos.
"""
import glob
import os
import re
import numpy as np
import cv2
import torch
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader

class STVENLoader(BaseLoader):
    """The data loader for STVEN pre-training."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes an STVEN dataloader.
            Args:
                name(str): name of the dataloader.
                data_path(str): (ignored) path of a folder which stores raw video and bvp data.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, raw_data_path):
        """Returns data directories from the configured CRF datasets."""
        # raw_data_path arg is ignored, we use config_data.CRF_DATASETS
        
        if not hasattr(self.config_data, 'CRF_DATASETS'):
             raise ValueError("STVENLoader requires CRF_DATASETS in config.TRAIN.DATA")
        
        crf_datasets = self.config_data.CRF_DATASETS
        # Expecting crf_datasets to be a Dict[int, str] mapping CRF -> Path
        
        dirs = []
        for crf_level, dataset_path in crf_datasets.items():
            if not os.path.exists(dataset_path):
                print(f"Warning: Dataset path for CRF {crf_level} does not exist: {dataset_path}")
                continue
                
            data_dirs = glob.glob(dataset_path + os.sep + "subject*") # returns a list of subjects' dir
            if not data_dirs:
                print(f"Warning: No subjects found in {dataset_path}")
                continue
            
            for data_dir in data_dirs:
                subject_match = re.search('subject(\d+)', data_dir)
                if subject_match:
                    subject_index = subject_match.group(1)
                    dirs.append({
                        "index": subject_index, 
                        "path": data_dir,
                        "crf": crf_level
                    })
        
        if not dirs:
             raise ValueError(self.dataset_name + " data paths empty!")

        # Sort by subject index to ensure all CRF levels for the same subject stay together
        # This prevents data leakage where subject X (CRF 0) is in train and subject X (CRF 23) is in valid
        dirs.sort(key=lambda x: int(x["index"]))
             
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values.
           Ensures that all CRF levels for the same subject stay in the same split.
        """
        if begin == 0 and end == 1:
            return data_dirs

        # Get unique subjects in original order (already sorted by subject index)
        unique_subjects = []
        for d in data_dirs:
            if d['index'] not in unique_subjects:
                unique_subjects.append(d['index'])
        
        num_subjects = len(unique_subjects)
        start_idx = int(begin * num_subjects)
        end_idx = int(end * num_subjects)
        
        chosen_subjects = unique_subjects[start_idx:end_idx]
        
        data_dirs_new = [d for d in data_dirs if d['index'] in chosen_subjects]
        return data_dirs_new

    def build_file_list_retroactive(self, data_dirs, begin, end):
        """Builds a list of files used by the dataloader for the data split.
           Overridden to handle STVEN's _crf{level} naming convention.
        """
        # get data split based on begin and end indices.
        data_dirs_subset = self.split_raw_data(data_dirs, begin, end)

        # generate a list of unique raw-data file names
        filename_list = []
        for i in range(len(data_dirs_subset)):
            filename_list.append(data_dirs_subset[i]['index'])
        filename_list = list(set(filename_list))  # ensure all indexes are unique

        # generate a list of all preprocessed / chunked data files
        file_list = []
        for fname in filename_list:
            # Glob for any CRF level for this subject
            processed_file_data = list(glob.glob(self.cached_path + os.sep + "{0}_crf*_input*.npy".format(fname)))
            file_list += processed_file_data

        if not file_list:
            raise ValueError(self.dataset_name,
                             'STVENLoader: File list empty. Check preprocessed data folder exists and is not empty. '
                             f'Path: {self.cached_path} searching for: {filename_list}')

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)  # save file list to .csv

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Invoke by preprocess_dataset for multi_process."""
        try:
            # We explicitly use the naming convention: {subject}_crf{crf}
            subject_name = data_dirs[i]['index']
            crf_level = data_dirs[i]['crf']
            saved_filename = f"{subject_name}_crf{crf_level}"
            
            # Assuming 'vid.mp4' is the standard name inside each specific dataset folder
            video_path = os.path.join(data_dirs[i]['path'], "vid.mp4")
            
            # Read Frames
            if 'None' in config_preprocess.DATA_AUG:
                 frames = self.read_video(video_path)
            elif 'Motion' in config_preprocess.DATA_AUG:
                 frames = self.read_npy_video(glob.glob(os.path.join(data_dirs[i]['path'], '*.npy')))
            else:
                 raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name}! Received {config_preprocess.DATA_AUG}.')

            # Generate Dummy Labels (Since this is unsupervised/self-supervised pre-training)
            # We create dummy BVP signals of the same length as frames
            # This is to satisfy the BaseLoader interface
            
            # Note: frame rate fs is needed for length calculation if we were doing real labels, 
            # but for dummy we just match frames length
            num_frames = len(frames)
            bvps = np.zeros(num_frames) 

            # Preprocess (Chunking, etc.)
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            
            input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
            file_list_dict[i] = input_name_list
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing {data_dirs[i]['path']}: {e}")

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)
        
    def __getitem__(self, index):
        """Returns a triplet: (compressed_video, uncompressed_video, bitrate_label)."""
        # Load compressed data (current item)
        compressed_data = np.load(self.inputs[index])
        
        # Handle Data Format
        if self.data_format == 'NDCHW':
            compressed_data = np.transpose(compressed_data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            compressed_data = np.transpose(compressed_data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')
        
        compressed_data = np.float32(compressed_data)

        # Identify filename info
        item_path = self.inputs[index]
        item_path_filename = item_path.split(os.sep)[-1]
        
        # Parse filename: e.g., "subject1_crf23_input0.npy"
        # We need to extract the CRF level
        try:
            # Regex to find crf patterns
            match = re.search(r'_crf(\d+)_', item_path_filename)
            if match:
                current_crf = int(match.group(1))
            else:
                 # If no crf found, assume 0 (uncompressed) if naming didn't follow convention, 
                 # but our preprocess enforces it. Fallback/Error.
                 raise ValueError(f"Could not parse CRF level from filename: {item_path_filename}")
        except Exception as e:
            print(f"Error parsing filename {item_path_filename}: {e}")
            raise e

        # Load Uncompressed Pair (CRF 0)
        # We assume the uncompressed file exists with the same chunk index/subject but 'crf0'
        # Replace _crf{current_crf}_ with _crf0_
        uncompressed_filename_base = item_path_filename.replace(f"_crf{current_crf}_", "_crf0_")
        uncompressed_path = os.path.join(self.cached_path, uncompressed_filename_base)
        
        if not os.path.exists(uncompressed_path):
             # Try to find it in case of path differences or if current IS 0
             if current_crf == 0:
                 uncompressed_data = compressed_data.copy()
             else:
                 # Ensure CRF 0 data was preprocessed!
                 raise FileNotFoundError(f"Corresponding uncompressed file not found: {uncompressed_path}")
        else:
             uncompressed_data = np.load(uncompressed_path)
             if self.data_format == 'NDCHW':
                uncompressed_data = np.transpose(uncompressed_data, (0, 3, 1, 2))
             elif self.data_format == 'NCDHW':
                uncompressed_data = np.transpose(uncompressed_data, (3, 0, 1, 2))
             uncompressed_data = np.float32(uncompressed_data)

        # Generate Label
        # One-hot encoding based on configured CRF levels
        crf_levels = self.config_data.CRF_LEVELS # Expecting List[int] e.g. [0, 23, 34]
        if current_crf not in crf_levels:
             # Should not happen if config is consistent
             raise ValueError(f"CRF {current_crf} found in file but not in config.CRF_LEVELS: {crf_levels}")
        
        label_idx = crf_levels.index(current_crf)
        bitrate_label = np.zeros(len(crf_levels), dtype=np.float32)
        bitrate_label[label_idx] = 1.0

        # Note: We return 3 items, unlike BaseLoader which returns 4 (data, label, filename, chunk_id)
        # The Trainer needs to be aware of this change or we stick to standard signature and pack it differently.
        # Given STVENTrainer expects (compressed, uncompressed, label), we return exactly that.
        
        return compressed_data, uncompressed_data, bitrate_label
