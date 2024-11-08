import logging
import os.path
import subprocess
import time
import typing
import glob
from typing import List
from leaderboards.mail_io import VLINCSMail
from leaderboards.dataset import Dataset
from leaderboards.test_harness_config import TestHarnessConfig


def check_gpu(host):
    if host == Task.LOCAL_VM_IP:
        child = subprocess.Popen(['nvidia-smi'])
    else:
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

    if host == Task.LOCAL_VM_IP:
        all_files = glob.glob('{}/*'.format(remote_scratch))
        child = subprocess.Popen(['rm', '-rf'] + all_files)
    else:
        child = subprocess.Popen(['ssh', '-q', 'vlincs@'+host, 'rm', '-rf', '{}/*'.format(remote_scratch)])
    return child.wait()


def create_directory_on_vm(host, dirpath: str):
    if host == Task.LOCAL_VM_IP:
        params = ['mkdir', '-p', dirpath]
    else:
        params = ['ssh', '-q', 'vlincs@' + host, 'mkdir', '-p', dirpath]
    child = subprocess.Popen(params)
    return child.wait()

def rsync_file_to_vm(host, source_filepath, remote_path, source_params = [], remote_params = []):
    params = []
    if host == Task.LOCAL_VM_IP:
        params.extend(['rsync'])
    else:
        params.extend(['rsync', '-e', 'ssh -q'])

    params.extend(source_params)
    if host == Task.LOCAL_VM_IP:
        params.extend([source_filepath, remote_path])
    else:
        params.extend([source_filepath, 'vlincs@' + host + ':' + remote_path])
    params.extend(remote_params)

    logging.debug(' '.join(params))

    rc = subprocess.run(params)
    return rc.returncode


def rsync_dir_to_vm(host, source_dirpath, remote_dirpath, source_params = [], remote_params = []):
    params = []
    if host == Task.LOCAL_VM_IP:
        params.extend(['rsync', '-ar', '--prune-empty-dirs', '--delete'])
    else:
        params.extend(['rsync', '-ar', '-e', 'ssh -q', '--prune-empty-dirs', '--delete'])
    params.extend(source_params)

    if host == Task.LOCAL_VM_IP:
        import shlex
        params.extend([shlex.quote(source_dirpath), shlex.quote(remote_dirpath)])
    else:
        params.extend([source_dirpath, 'vlincs@' + host + ':' + remote_dirpath])
    params.extend(remote_params)

    logging.debug(' '.join(params))

    rc = subprocess.run(params)
    return rc.returncode


def scp_dir_from_vm(host, remote_dirpath, local_dirpath):
    logging.debug('remote: {} to {}'.format(remote_dirpath, local_dirpath))
    if host == Task.LOCAL_VM_IP:
        cmd = ['cp', '-r'] + glob.glob('{}/*'.format(remote_dirpath)) + [local_dirpath]
        # child = subprocess.Popen(cmd)
    else:
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
    LOCAL_VM_IP = 'local'

    def __init__(self):
        pass

    def check_instance_params(self, test_harness_config: TestHarnessConfig):
        raise NotImplementedError()

    def get_remote_dataset_dirpath(self, remote_dirpath, leaderboard_name):
        raise NotImplementedError()

    def verify_dataset(self, leaderboard_name, dataset: Dataset, required_files: List[str]):
        raise NotImplementedError()

    def run_basic_checks(self, vm_ip, vm_name):
        raise NotImplementedError()

    def run_submission_checks(self, submission_filepath):
        raise NotImplementedError()

    def run_submission_schema_header_checks(self, submission_filepath):
        raise NotImplementedError()

    def copy_in_env(self, vm_ip, vm_name, test_harness_config: TestHarnessConfig, custom_remote_home: str=None, custom_remote_scratch: str=None):
        raise NotImplementedError()

    def copy_in_task_data(self, vm_ip, vm_name, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str], custom_remote_home: str=None, custom_remote_scratch: str=None, custom_metaparameter_filepath: str=None):
        raise NotImplementedError()

    def execute_submission(self, vm_ip, vm_name, python_execution_env_filepath: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str], info_dict: dict, custom_remote_home: str=None, custom_remote_scratch: str=None, custom_metaparameter_filepath: str=None, subset_model_ids: list=None, custom_result_dirpath: str=None):
        raise NotImplementedError()

    def get_basic_execute_args(self, vm_ip: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, excluded_files: List[str],  custom_remote_home: str, custom_remote_scratch: str , custom_metaparameter_filepath: str, subset_model_ids: list, custom_result_dirpath: str):
        raise NotImplementedError()

    def get_custom_execute_args(self, vm_ip: str, submission_filepath: str, dataset: Dataset, training_dataset: Dataset, custom_remote_home: str, custom_remote_scratch: str, custom_result_dirpath: str):
        raise NotImplementedError()

    def copy_out_results(self, vm_ip, vm_name, result_dirpath, custom_remote_home: str=None, custom_remote_scratch: str=None):
        raise NotImplementedError()

    def package_results(self, result_dirpath: str, info_dict: dict):
        raise NotImplementedError()

    def cleanup_vm(self, vm_ip, vm_name, custom_remote_home: str=None, custom_remote_scratch: str=None):
        raise NotImplementedError()

    def load_ground_truth(self, dataset: Dataset) -> typing.OrderedDict[str, float]:
        raise NotImplementedError()

