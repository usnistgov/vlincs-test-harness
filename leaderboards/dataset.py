from leaderboards.test_harness_config import TestHarnessConfig
import os
import copy

class Dataset(object):
    BUFFER_TIME = 900
    # 300 models, 3 minutes per model + 15 minute buffer
    DEFAULT_TIMEOUT_SEC = 600 * 300 + BUFFER_TIME
    DEFAULT_STS_TIMEOUT_SEC = 600 * 10 + BUFFER_TIME
    DATASET_SUFFIX = 'dataset'
    # DATASET_GROUNDTRUTH_NAME = 'groundtruth'
    MODEL_DIRNAME = 'models'
    SOURCE_DATA_NAME = 'source-data'
    METADATA_NAME = 'METADATA.csv'
    GROUND_TRUTH_NAME = 'ground_truth.csv'

    def __init__(self, test_harness_config: TestHarnessConfig,
                 leaderboard_name: str,
                 split_name: str,
                 can_submit: bool,
                 slurm_queue_name: str,
                 slurm_nice: int,
                 has_source_data: bool,
                 timeout_time_per_model_sec: int=600,
                 auto_delete_submission: bool=False,
                 auto_execute_split_names=None):

        self.split_name = split_name
        self.dataset_name = self.get_dataset_name()
        self.dataset_dirpath: str = os.path.join(test_harness_config.datasets_dirpath, leaderboard_name, self.dataset_name)
        self.results_dirpath = os.path.join(test_harness_config.results_dirpath, '{}-dataset'.format(leaderboard_name), self.dataset_name)
        self.can_submit = can_submit
        self.slurm_queue_name = slurm_queue_name
        self.slurm_nice = slurm_nice
        # self.excluded_files = excluded_files
        # self.required_files = required_files
        # self.submission_metrics

        self.source_dataset_dirpath = None

        if has_source_data:
            self.source_dataset_dirpath = os.path.join(test_harness_config.datasets_dirpath, leaderboard_name, '{}'.format(Dataset.SOURCE_DATA_NAME))

        self.auto_delete_submission = auto_delete_submission
        self.auto_execute_split_names = []

        if auto_execute_split_names is not None:
            for split in auto_execute_split_names:
                self.auto_execute_split_names.append(split)

        num_models = self.get_num_models()

        self.submission_window_time_sec = Dataset.BUFFER_TIME
        if num_models > 0:
            self.timeout_time_sec = num_models * timeout_time_per_model_sec
        else:
            if self.split_name == 'sts':
                self.timeout_time_sec = Dataset.DEFAULT_STS_TIMEOUT_SEC
            else:
                self.timeout_time_sec = Dataset.DEFAULT_TIMEOUT_SEC

    def get_num_models(self):
        model_dirpath = os.path.join(self.dataset_dirpath, Dataset.MODEL_DIRNAME)
        if os.path.exists(model_dirpath):
            return len([name for name in os.listdir(model_dirpath) if os.path.isdir(os.path.join(model_dirpath, name))])
        else:
            return 0

    def get_dataset_name(self):
        return '{}-{}'.format(self.split_name, Dataset.DATASET_SUFFIX)

    def initialize_directories(self):
        os.makedirs(self.dataset_dirpath, exist_ok=True)
        os.makedirs(self.results_dirpath, exist_ok=True)

    def __str__(self):
        msg = "Dataset: \n"
        for key, value in self.__dict__.items():
            msg += '\t{} = {}\n'.format(key, value)
        msg += ')'
        return msg

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

        self.datasets[dataset.split_name] = dataset
        dataset.initialize_directories()
        print('Created: {}'.format(dataset))

    def get(self, split_name) -> Dataset:
        if split_name in self.datasets.keys():
            return self.datasets[split_name]
        else:
            raise RuntimeError('Invalid key in DatasetManager: {}'.format(split_name))

    def initialize_directories(self):
        for dataset in self.datasets.values():
            dataset.initialize_directories()
