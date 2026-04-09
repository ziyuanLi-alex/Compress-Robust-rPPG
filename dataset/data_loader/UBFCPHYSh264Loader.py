"""The dataloader for UBFC-PHYS H.264-compressed videos.

Details for the UBFC-PHYS Dataset see https://sites.google.com/view/ybenezeth/ubfc-phys.
If you use this dataset, please cite this paper:
R. Meziati Sabour, Y. Benezeth, P. De Oliveira, J. Chappé, F. Yang.
"UBFC-Phys: A Multimodal Database For Psychophysiological Studies Of Social Stress",
IEEE Transactions on Affective Computing, 2021.
"""

import glob
import os
import re
import gc
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import csv
import pandas as pd


class UBFCPHYSh264Loader(BaseLoader):
    """The data loader for UBFC-PHYS H.264-compressed dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes an UBFC-PHYS h264 dataloader.
        Args:
            data_path(str): path of a folder which stores compressed video and bvp data.
            e.g. data_path should be "UBFC-Phys-CRF14/RawData" for:
            -----------------
                 RawData/
                 |   |-- s1/
                 |       |-- vid_s1_T1.mp4
                 |       |-- vid_s1_T2.mp4
                 |       |-- vid_s1_T3.mp4
                 |       |-- bvp_s1_T1.csv
                 |       |-- bvp_s1_T2.csv
                 |       |-- bvp_s1_T3.csv
                 |   |-- s2/
                 |       |-- vid_s2_T1.mp4
                 |       |-- ...
            -----------------
            name(string): name of the dataloader.
            config_data(CfgNode): data settings(ref:config.py).
        """
        self.filtering = config_data.FILTERING
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-PHYS h264 dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "s*" + os.sep + "*.mp4")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [
            {"index": re.search("vid_(.*).mp4", data_dir).group(1), "path": data_dir}
            for data_dir in data_dirs
        ]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(
        self, data_dirs, config_preprocess, i, file_list_dict
    ):
        """invoked by preprocess_dataset for multi_process.

        Uses streaming to read raw frames, processes each streaming chunk
        (crop+resize+transform) to keep memory low, accumulates the small
        processed frames, then applies the same chunking and saving logic
        as the original loader for consistent output.
        """
        saved_filename = data_dirs[i]["index"]
        video_path = os.path.join(data_dirs[i]["path"])

        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            frames = self.read_video(video_path)
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
            bvps = BaseLoader.resample_ppg(bvps, frames.shape[0])
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            input_name_list, label_name_list = self.save_multi_process(
                frames_clips, bvps_clips, saved_filename
            )
            file_list_dict[i] = input_name_list
            del frames, bvps, frames_clips, bvps_clips
            gc.collect()
            return

        bvp_path = os.path.join(
            os.path.dirname(data_dirs[i]["path"]),
            "bvp_{0}.csv".format(saved_filename),
        )
        bvps_raw = self.read_wave(bvp_path)

        processed_data_list = []
        for frames_chunk in self.read_video_streaming(video_path, chunk_length=300):
            frames_chunk = self.crop_face_resize(
                frames_chunk,
                config_preprocess.CROP_FACE.DO_CROP_FACE,
                config_preprocess.CROP_FACE.BACKEND,
                config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
                config_preprocess.CROP_FACE.LARGE_BOX_COEF,
                config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
                config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
                config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
                config_preprocess.RESIZE.W,
                config_preprocess.RESIZE.H,
            )

            data_list = []
            for data_type in config_preprocess.DATA_TYPE:
                f_c = frames_chunk.copy()
                if data_type == "Raw":
                    data_list.append(f_c)
                elif data_type == "DiffNormalized":
                    data_list.append(BaseLoader.diff_normalize_data(f_c))
                elif data_type == "Standardized":
                    data_list.append(BaseLoader.standardized_data(f_c))
                else:
                    raise ValueError("Unsupported data type!")
            data = np.concatenate(data_list, axis=-1)
            processed_data_list.append(data)

            del frames_chunk, data_list, data
            gc.collect()

        if not processed_data_list:
            file_list_dict[i] = []
            return

        data_all = np.concatenate(processed_data_list, axis=0)
        actual_frame_count = data_all.shape[0]
        del processed_data_list
        gc.collect()

        bvps = BaseLoader.resample_ppg(bvps_raw, actual_frame_count)
        del bvps_raw
        gc.collect()

        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = BaseLoader.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoader.standardized_label(bvps)
        else:
            raise ValueError("Unsupported label type!")

        if config_preprocess.DO_CHUNK:
            frames_clips, bvps_clips = self.chunk(
                data_all, bvps, config_preprocess.CHUNK_LENGTH
            )
        else:
            frames_clips = np.array([data_all])
            bvps_clips = np.array([bvps])

        input_name_list, label_name_list = self.save_multi_process(
            frames_clips, bvps_clips, saved_filename
        )
        file_list_dict[i] = input_name_list

        del data_all, bvps, frames_clips, bvps_clips
        gc.collect()

    def load_preprocessed_data(self):
        """Loads the preprocessed data listed in the file list."""
        file_list_path = self.file_list_path
        file_list_df = pd.read_csv(file_list_path)
        base_inputs = file_list_df["input_files"].tolist()
        filtered_inputs = []

        for input in base_inputs:
            input_name = input.split(os.sep)[-1].split(".")[0].rsplit("_input", 1)[0]
            if (
                self.filtering.USE_EXCLUSION_LIST
                and input_name in self.filtering.EXCLUSION_LIST
            ):
                continue
            if self.filtering.SELECT_TASKS and not any(
                task in input_name for task in self.filtering.TASK_LIST
            ):
                continue
            filtered_inputs.append(input)

        if not filtered_inputs:
            raise ValueError(self.dataset_name + " dataset loading data error!")

        filtered_inputs = sorted(filtered_inputs)
        labels = [
            input_file.replace("input", "label") for input_file in filtered_inputs
        ]
        self.inputs = filtered_inputs
        self.labels = labels
        self.preprocessed_data_len = len(filtered_inputs)

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3)

        Note: This method loads all frames into memory. Use read_video_streaming
        for memory-efficient processing of large videos.
        """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        VidObj.release()
        return np.asarray(frames)

    def read_video_streaming(self, video_file, chunk_length=300):
        """Streams video frames in chunks to reduce memory usage.

        Args:
            video_file(str): path to video file
            chunk_length(int): number of frames per chunk

        Yields:
            np.array: chunks of frames with shape (chunk_length, H, W, 3)
        """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)

        frames_buffer = []
        success, frame = VidObj.read()

        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames_buffer.append(frame)

            # Yield chunk when buffer is full
            if len(frames_buffer) >= chunk_length:
                yield np.asarray(frames_buffer)
                frames_buffer = []  # Free memory

            success, frame = VidObj.read()

        VidObj.release()

        # Yield remaining frames
        if frames_buffer:
            yield np.asarray(frames_buffer)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        bvp = []
        with open(bvp_file, "r") as f:
            d = csv.reader(f)
            for row in d:
                bvp.append(float(row[0]))
        return np.asarray(bvp)
