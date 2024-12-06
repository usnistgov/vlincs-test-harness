import json
import logging
import os
import shutil
import subprocess
import typing
from typing import List
from mail_io import VLINCSMail
from dataset import Dataset
from test_harness_config import TestHarnessConfig
from metrics import Metric, VLINCSMetric


def check_gpu(host):
    child = subprocess.Popen(['ssh', '-q', 'vlincs@'+host, 'nvidia-smi'])
    return child.wait()


def check_file_in_container(container_filepath, filepath_in_container):
    child = subprocess.Popen(['singularity', 'exec', container_filepath, 'test', '-f', filepath_in_container])
    return child.wait()


def check_dir_in_container(container_filepath, dirpath_in_container):
    child = subprocess.Popen(['singularity', 'exec', container_filepath, 'test', '-d', dirpath_in_container])
    return child.wait()


def cleanup_scratch(host, remote_scratch):
    if remote_scratch == '':
        logging.error('Failed to cleanup scratch, errors with passing path: {}, it must not be an empty string'.format(remote_scratch))
        return -1

    child = subprocess.Popen(['ssh', '-q', 'vlincs@'+host, 'rm', '-rf', '{}/*'.format(remote_scratch)])
    return child.wait()


def create_directory_on_vm(host, dirpath: str):
    params = ['ssh', '-q', 'vlincs@' + host, 'mkdir', '-p', dirpath]
    child = subprocess.Popen(params)
    return child.wait()

def rsync_file_to_vm(host, source_filepath, remote_path, source_params = [], remote_params = []):
    params = []
    params.extend(['rsync', '-e', 'ssh -q'])

    params.extend(source_params)
    params.extend([source_filepath, 'vlincs@' + host + ':' + remote_path])
    params.extend(remote_params)

    logging.debug(' '.join(params))

    rc = subprocess.run(params)
    return rc.returncode


def rsync_dir_to_vm(host, source_dirpath, remote_dirpath, source_params = [], remote_params = []):
    params = []
    params.extend(['rsync', '-ar', '-e', 'ssh -q', '--prune-empty-dirs', '--delete'])
    params.extend(source_params)

    params.extend([source_dirpath, 'vlincs@' + host + ':' + remote_dirpath])
    params.extend(remote_params)

    logging.debug(' '.join(params))

    rc = subprocess.run(params)
    return rc.returncode


def scp_dir_from_vm(host, remote_dirpath, local_dirpath):
    logging.debug('remote: {} to {}'.format(remote_dirpath, local_dirpath))
    cmd = ['scp', '-r', '-q', 'vlincs@{}:{}/*'.format(host, remote_dirpath), local_dirpath]
        # child = subprocess.Popen(cmd)
    logging.info(' '.join(cmd))
    rc = subprocess.run(cmd)
    return rc.returncode
    # return child.wait()


def check_subprocess_error(sc, errors, msg, send_mail=False, subject=''):
    if sc != 0:
        message = '{}, status code: {}'.format(msg, sc)
        logging.error(message)

        if send_mail:
            VLINCSMail().send(to='vlincs@nist.gov', subject=subject, message=message)

        return errors

    return ''


class Task(object):

    METRIC_RESULT_FILENAME = 'metric_results.json'

    def __init__(self):
        pass

    def check_instance_params(self, test_harness_config: TestHarnessConfig):
        raise NotImplementedError()

    def get_remote_dataset_dirpath(self, remote_dirpath, leaderboard_name):
        raise NotImplementedError()

    def run_basic_checks(self, vm_ip, vm_name):
        raise NotImplementedError()

    def run_submission_checks(self, submission_filepath, dataset: Dataset):
        raise NotImplementedError()

    def copy_in_env(self, vm_ip, vm_name, test_harness_config: TestHarnessConfig):
        raise NotImplementedError()

    def copy_in_task_data(self, vm_ip, vm_name, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str]):
        raise NotImplementedError()

    def execute_submission(self, vm_ip, vm_name, python_execution_env_filepath: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str], info_dict: dict, results_dirpath: str):
        raise NotImplementedError()

    def get_basic_execute_args(self, vm_ip: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str]):
        raise NotImplementedError()

    def get_custom_execute_args(self, vm_ip: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset):
        raise NotImplementedError()

    def copy_out_results(self, vm_ip, vm_name, results_dirpath):
        raise NotImplementedError()

    def cleanup_vm(self, vm_ip, vm_name):
        raise NotImplementedError()

    def process_metrics(self, results_dirpath: str, dataset: Dataset, metrics: typing.Dict[str, Metric], actor_name: str, leaderboard_name: str):
        raise NotImplementedError()


class TakeHomeTask(Task):
    def __init__(self, test_harness_config: TestHarnessConfig, leaderboard_name: str):
        super().__init__()

    def check_instance_params(self, test_harness_config: TestHarnessConfig):
        has_updated = False
        return has_updated

    def get_remote_dataset_dirpath(self, remote_dirpath, leaderboard_name):
        pass

    def run_basic_checks(self, vm_ip, vm_name):
        errors = ''
        return errors

    def run_submission_checks(self, submission_filepath, dataset: Dataset):
        errors = ''
        return errors

    def copy_in_env(self, vm_ip, vm_name, test_harness_config: TestHarnessConfig):
        errors = ''
        return errors

    def copy_in_task_data(self, vm_ip, vm_name, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str]):
        errors = ''
        return errors

    def execute_submission(self, vm_ip, vm_name, python_execution_env_filepath: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str], info_dict: dict, results_dirpath: str):
        errors = ''
        shutil.unpack_archive(submission_filepath, results_dirpath)

        if not dataset.verify_results(results_dirpath):
            errors += ':Missing Results:'

        return errors

    def get_basic_execute_args(self, vm_ip: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str]):
        return []

    def get_custom_execute_args(self, vm_ip: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset):
        return []

    def copy_out_results(self, vm_ip, vm_name, results_dirpath):
        errors = ''
        return errors

    def cleanup_vm(self, vm_ip, vm_name):
        errors = ''
        return errors

    def process_metrics(self, results_dirpath: str, dataset: Dataset, metrics: typing.Dict[str, Metric], actor_name: str, leaderboard_name: str):
        results = dataset.load_results(results_dirpath)
        ground_truth = dataset.load_ground_truth()
        metadata_df = None

        all_results = {}

        for metric_name, metric in metrics.items():
            if isinstance(metric, VLINCSMetric):
                metric_output = metric.compute(results, ground_truth, metadata_df, actor_name, leaderboard_name, dataset.split_name, results_dirpath)
                all_results[metric_name] = metric_output

        metric_result_filepath = os.path.join(results_dirpath, Task.METRIC_RESULT_FILENAME)
        with open(metric_result_filepath, 'w') as fp:
            json.dump(all_results, fp, indent=4)
