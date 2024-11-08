import argparse
import json
import sys
import os
import random
import shutil
import subprocess
import logging
import glob
import time

import signal
from spython.main import Client

from abc import ABC, abstractmethod


class TimeoutError(Exception):
    pass

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def rsync_dirpath(source_dirpath: str, dest_dirpath: str, rsync_args: list):
    params = ['rsync']
    params.extend(rsync_args)
    params.extend(glob.glob(source_dirpath))
    params.append(dest_dirpath)

    child = subprocess.Popen(params)
    return child.wait()

def clean_dirpath_contents(dirpath: str):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            if os.path.isfile(filepath) or os.path.islink(filepath):
                os.unlink(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
        except Exception as e:
            logging.info('Failed to delete {}, reason {}'.format(filepath, e))


class EvaluateTask(ABC):

    def __init__(self):
        pass

    def process_models(self):
        raise NotImplementedError('Must override execute_task')


    @abstractmethod
    def get_singularity_instance_options(self, active_dirpath, scratch_dirpath, uses_gpu=True):
        raise NotImplementedError('Must override execute_task')


    @abstractmethod
    def get_execute_task_args(self, active_dirpath: str, container_scratch_dirpath: str, active_result_filepath: str):
        raise NotImplementedError('Must override execute_task')


if __name__ == '__main__':
    VALID_TASK_TYPES = {
    }

    parser = argparse.ArgumentParser(description='Entry point to execute containers')

    parser.add_argument('--timeout', type=int, help='The amount of time to timeout the execution in seconds')
    parser.add_argument('--models-dirpath',  type=str, help='The directory path to models to evaluate', required=True)
    parser.add_argument('--task-type', type=str, choices=VALID_TASK_TYPES.keys(), help='The type of submission', required=True)
    parser.add_argument('--submission-filepath', type=str, help='The filepath to the submission', required=True)
    parser.add_argument('--home-dirpath', type=str, help='The directory path to home', required=True)
    parser.add_argument('--result-dirpath', type=str, help='The directory path for results', required=True)
    parser.add_argument('--scratch-dirpath', type=str, help='The directory path for scratch', required=True)
    parser.add_argument('--training-dataset-dirpath', type=str, help='The directory path to the training dataset', required=False, default=None)
    parser.add_argument('--metaparameter-filepath', type=str, help='The directory path for the metaparameters file when running custom metaparameters', required=False, default=None)
    parser.add_argument('--rsync-excludes', nargs='*', help='List of files to exclude for rsyncing data', required=False, default=None)
    parser.add_argument('--learned-parameters-dirpath', type=str, help='The directory path to the learned parameters', required=False, default=None)
    parser.add_argument('--source-dataset-dirpath', type=str, help='The source dataset directory path', required=False, default=None)
    parser.add_argument('--result-prefix-filename', type=str, help='The prefix name given to results', required=False, default=None)
    parser.add_argument('--subset-model-ids', nargs='*', help='List of model IDs to evaluate on', required=False, default=None)

    args, extras = parser.parse_known_args()

    task_type = args.task_type

    evaluate_task_instance = VALID_TASK_TYPES[task_type](models_dirpath=args.models_dirpath,
                                                         submission_filepath=args.submission_filepath,
                                                         home_dirpath=args.home_dirpath,
                                                         result_dirpath=args.result_dirpath,
                                                         scratch_dirpath=args.scratch_dirpath,
                                                         training_dataset_dirpath=args.training_dataset_dirpath,
                                                         metaparameters_filepath=args.metaparameter_filepath,
                                                         rsync_excludes=args.rsync_excludes,
                                                         learned_parameters_dirpath=args.learned_parameters_dirpath,
                                                         source_dataset_dirpath=args.source_dataset_dirpath,
                                                         result_prefix_filename=args.result_prefix_filename,
                                                         subset_model_ids=args.subset_model_ids,
                                                         timeout_time=args.timeout)

    try:
        evaluate_task_instance.process_models()
    except TimeoutError as e:
        logging.info('Your submission has failed to process all models within the timelimit ({}s)'.format(args.timeout))
        exit(-9)
