import configparser
import csv
import logging
import typing

from leaderboards.test_harness_config import TestHarnessConfig
import os
import copy
import pandas as pd

class Dataset(object):
    BUFFER_TIME = 900
    DEFAULT_TIMEOUT_SEC = 600 * 300 + BUFFER_TIME
    DEFAULT_STS_TIMEOUT_SEC = 600 * 10 + BUFFER_TIME
    METADATA_NAME = 'METADATA.csv'

    def __init__(self, test_harness_config: TestHarnessConfig,
                 leaderboard_name: str,
                 split_name: str,
                 can_submit: bool,
                 slurm_queue_name: str,
                 slurm_nice: int,
                 timeout_time_per_model_sec: int=600,
                 auto_delete_submission: bool=False,
                 auto_execute_split_names=None):

        self.leaderboard_name = leaderboard_name
        self.split_name = split_name
        self.dataset_name = self.get_dataset_name()
        self.dataset_dirpath: str = os.path.join(test_harness_config.datasets_dirpath, leaderboard_name, self.dataset_name)
        self.results_dirpath = os.path.join(test_harness_config.results_dirpath, '{}'.format(leaderboard_name), self.dataset_name)
        self.can_submit = can_submit
        self.slurm_queue_name = slurm_queue_name
        self.slurm_nice = slurm_nice

        self.auto_delete_submission = auto_delete_submission
        self.auto_execute_split_names = []

        if auto_execute_split_names is not None:
            for split in auto_execute_split_names:
                self.auto_execute_split_names.append(split)

        num_data = len(self)

        self.submission_window_time_sec = Dataset.BUFFER_TIME
        if num_data > 0:
            self.timeout_time_sec = num_data * timeout_time_per_model_sec
        else:
            if self.split_name == 'sts':
                self.timeout_time_sec = Dataset.DEFAULT_STS_TIMEOUT_SEC
            else:
                self.timeout_time_sec = Dataset.DEFAULT_TIMEOUT_SEC

    def initialize_directories(self):
        os.makedirs(self.dataset_dirpath, exist_ok=True)
        os.makedirs(self.results_dirpath, exist_ok=True)

    def __str__(self):
        msg = "Dataset: \n"
        for key, value in self.__dict__.items():
            msg += '\t{} = {}\n'.format(key, value)
        msg += ')'
        return msg

    def get_dataset_name(self):
        return '{}'.format(self.split_name)

    def verify(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def load_ground_truth(self):
        raise NotImplementedError()

    def verify_results(self, results_dirpath):
        raise NotImplementedError()

    def load_results(self, results_dirpath) -> typing.Dict[str, pd.DataFrame | None]:
        raise NotImplementedError()


class VideoLINCSDataset(Dataset):
    SEQMAP_NAME = 'seqmap.txt'
    GROUNDTRUTH_NAME = 'gt.txt'
    SEQINFO_NAME = 'seqinfo.ini'

    GROUND_TRUTH_COLUMNS = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility']

    def __init__(self, test_harness_config: TestHarnessConfig,
                 leaderboard_name: str,
                 split_name: str,
                 can_submit: bool,
                 slurm_queue_name: str,
                 slurm_nice: int,
                 timeout_time_per_model_sec: int=600,
                 auto_delete_submission: bool=False,
                 auto_execute_split_names=None):
        self.leaderboard_name = leaderboard_name
        self.split_name = split_name
        self.dataset_name = self.get_dataset_name()
        self.dataset_dirpath: str = os.path.join(test_harness_config.datasets_dirpath, leaderboard_name, self.dataset_name)

        self.seqmap_filepath = os.path.join(self.dataset_dirpath, VideoLINCSDataset.SEQMAP_NAME)
        self.video_names = []

        if os.path.exists(self.seqmap_filepath):
            with open(self.seqmap_filepath, 'r') as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    if i == 0 or row[0] == '':
                        continue

                    video_name = row[0]
                    self.video_names.append(video_name)

        super().__init__(test_harness_config, leaderboard_name, split_name, can_submit, slurm_queue_name, slurm_nice, timeout_time_per_model_sec, auto_delete_submission, auto_execute_split_names)


    def verify(self):
        logging.info('Verifying dataset {} for leaderboard {}'.format(self.dataset_name, self.leaderboard_name))
        if not os.path.exists(self.dataset_dirpath):
            logging.warning('Failed to find directory {} for'.format(self.dataset_dirpath))
            return False

        if not os.path.exists(self.seqmap_filepath):
            logging.warning('Failed to find seqmap file {}'.format(self.seqmap_filepath))
            return False

        passed_video_check = True

        for video_name in self.video_names:
            video_dirpath = os.path.join(self.dataset_dirpath, video_name)

            # TODO: Add checks for video files
            if os.path.exists(video_dirpath):
                # check for ground truth and seqinfo
                gt_filepath = os.path.join(video_dirpath, VideoLINCSDataset.GROUNDTRUTH_NAME)
                if not os.path.exists(gt_filepath):
                    logging.warning('Failed to find gt for video {} at {}'.format(video_name, gt_filepath))
                    passed_video_check = False
                seqinfo_filepath = os.path.join(video_dirpath, VideoLINCSDataset.SEQINFO_NAME)
                if not os.path.exists(seqinfo_filepath):
                    logging.warning('Failed to find seqinfo for video {} at {}'.format(video_name, seqinfo_filepath))
                    passed_video_check = False
            else:
                passed_video_check = False

            return passed_video_check

    def verify_results(self, results_dirpath):
        passed_check = True
        for video_name in self.video_names:
            video_result_filepath = os.path.join(results_dirpath, '{}.txt'.format(video_name))
            if not os.path.exists(video_result_filepath):
                logging.warning('Failed to find result file: {}'.format(video_result_filepath))
                passed_check = False

        return passed_check

    def load_results(self, results_dirpath) -> typing.Dict[str, pd.DataFrame | None]:
        results = {}
        for video_name in self.video_names:
            video_result_filepath = os.path.join(results_dirpath, '{}.txt'.format(video_name))
            if os.path.exists(video_result_filepath):
                results[video_name] = pd.read_csv(video_result_filepath, header=None, names=VideoLINCSDataset.GROUND_TRUTH_COLUMNS)
            else:
                results[video_name] = None
        return results

    def __len__(self):
        return len(self.video_names)

    def load_ground_truth(self):
        # Ground truth is for each video there are N frames, each frame we either initialize everything to no detections or having some detections
        gt_data = {}

        for video_name in self.video_names:
            gt_data[video_name] = {}
            video_dirpath = os.path.join(self.dataset_dirpath, video_name)
            seqinfo_filepath = os.path.join(video_dirpath, VideoLINCSDataset.SEQINFO_NAME)

            if not os.path.exists(seqinfo_filepath):
                logging.warning('Failed to find seqinfo file: {} for video {} on leaderboard {}'.format(seqinfo_filepath, video_name, self.leaderboard_name))
                raise Exception('Failed to find seqinfo file: {} for video {} on leaderboard {}'.format(seqinfo_filepath, video_name, self.leaderboard_name))

            ini_data = configparser.ConfigParser()
            ini_data.read(seqinfo_filepath)
            gt_data[video_name]['seqLength'] = int(ini_data['Sequence']['seqLength'])

            gt_filepath = os.path.join(video_dirpath, VideoLINCSDataset.GROUNDTRUTH_NAME)

            if not os.path.exists(gt_filepath):
                logging.warning('Failed to find gt file: {} for video {} on leaderboard {}'.format(gt_filepath, video_name,self.leaderboard_name))
                raise Exception('Failed to find gt file: {} for video {} on leaderboard {}'.format(gt_filepath, video_name, self.leaderboard_name))
            df = pd.read_csv(gt_filepath, header=None, names=VideoLINCSDataset.GROUND_TRUTH_COLUMNS)
            gt_data[video_name]['df'] = df

        return gt_data


class DatasetManager(object):
    def __init__(self):
        self.datasets = {}

    def __str__(self):
        msg = "Datasets: \n"
        for dataset in self.datasets.values():
            msg = msg + "  " + dataset.__str__() + "\n"
        return msg

    def can_submit_to_dataset(self, data_split_name: str):
        if data_split_name in self.datasets.keys():
            dataset = self.datasets[data_split_name]
            return dataset.can_submit
        return False

    def get_submission_dataset_split_names(self):
        result = []
        for data_split_name, dataset in self.datasets.items():
            if dataset.can_submit:
                result.append(data_split_name)

        return result

    def has_dataset(self, split_name: str):
        return split_name in self.datasets.keys()

    def add_dataset(self, dataset: Dataset):
        if dataset.split_name in self.datasets.keys():
            raise RuntimeError('Dataset already exists in DatasetManager: {}'.format(dataset.dataset_name))

        if dataset.verify():
            self.datasets[dataset.split_name] = dataset
            dataset.initialize_directories()
            print('Created: {}'.format(dataset))
            return True

        return False




    def get(self, split_name) -> Dataset:
        if split_name in self.datasets.keys():
            return self.datasets[split_name]
        else:
            raise RuntimeError('Invalid key in DatasetManager: {}'.format(split_name))

    def initialize_directories(self):
        for dataset in self.datasets.values():
            dataset.initialize_directories()
