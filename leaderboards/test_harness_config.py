import json
import os

from leaderboards import json_io
from python_utils import update_object_values
from typing import Dict

class TestHarnessConfig(object):

    CONFIG_FILENAME = 'test_harness_config.json'
    DEFAULT_VM_CONFIGURATION = {'gpu-vm-01': '192.168.200.2', 'gpu-vm-41': '192.168.200.3', 'gpu-vm-81': '192.168.200.4', 'gpu-vm-c1': '192.168.200.5'}

    def __init__(self, test_harness_dirpath: str, token_pickle_filepath: str, slurm_execute_script_filepath: str=None, init=False, control_slurm_queue_name='control'):
        self.test_harness_dirpath = os.path.abspath(test_harness_dirpath)
        self.test_harness_config_filepath = os.path.join(self.test_harness_dirpath, TestHarnessConfig.CONFIG_FILENAME)
        self.token_pickle_filepath = token_pickle_filepath
        self.html_repo_dirpath = os.path.join(self.test_harness_dirpath, 'html')
        self.submission_dirpath = os.path.join(self.test_harness_dirpath, 'submissions')
        self.datasets_dirpath = os.path.join(self.test_harness_dirpath, 'datasets')
        self.results_dirpath = os.path.join(self.test_harness_dirpath, 'results')
        self.leaderboard_configs_dirpath = os.path.join(self.test_harness_dirpath, 'leaderboard-configs')
        self.leaderboard_results_dirpath = os.path.join(self.test_harness_dirpath, 'leaderboard-results')
        self.actors_filepath = os.path.join(self.test_harness_dirpath, 'actors.json')
        self.log_filepath = os.path.join(self.test_harness_dirpath, 'test_harness.log')
        file_dirpath = os.path.dirname(os.path.realpath(__file__))
        self.root_test_harness_dirpath = os.path.normpath(os.path.join(file_dirpath, '..'))
        self.task_evaluator_script_filepath = os.path.join(file_dirpath, 'task_executor.py')
        self.scanner_script_filepath = os.path.join(file_dirpath, 'run_scan.sh')
        self.python_env = '/home/vlincs/vlincs-env/bin/python3'
        self.evaluate_python_env = '/home/vlincs/miniconda3/envs/vlincs_evaluate/bin/python'
        self.local_test_harness_conda_env = '/home/vlincs/miniconda3'
        self.leaderboard_csvs_dirpath = os.path.join(test_harness_dirpath, 'leaderboard-summary-csvs')
        self.vm_cpu_cores_per_partition = {'es': 10, 'sts': 10}
        self.job_color_key = {604800: 'text-success font-weight-bold',
                              1209600: 'text-warning font-weight-bold',
                              float('inf'): 'text-danger font-weight-bold'}

        self.summary_metric_email_addresses = []
        self.summary_metrics_dirpath = os.path.join(test_harness_dirpath, 'summary-metrics')
        self.summary_metric_update_timeframe = 3600
        self.last_summary_metric_update = 0

        self.slurm_execute_script_filepath = slurm_execute_script_filepath
        if slurm_execute_script_filepath is None:
            file_dirpath = os.path.dirname(os.path.realpath(__file__))
            slurm_scripts_dirpath = os.path.join(file_dirpath, '..', 'slurm_scripts')
            self.slurm_execute_script_filepath = os.path.normpath(os.path.join(slurm_scripts_dirpath, 'run_task.sh'))

        self.control_slurm_queue_name = control_slurm_queue_name
        self.commit_and_push_html = True
        self.accepting_submissions = False
        self.active_leaderboard_names = list()
        self.archive_leaderboard_names = list()
        self.html_default_leaderboard_name = ''
        self.vms = TestHarnessConfig.DEFAULT_VM_CONFIGURATION


        self.log_file_byte_limit = int(1 * 1024 * 1024)

        if init:
            self.initialize_directories()


    def initialize_directories(self):
        os.makedirs(self.test_harness_dirpath, exist_ok=True)
        os.makedirs(self.html_repo_dirpath, exist_ok=True)
        os.makedirs(self.submission_dirpath, exist_ok=True)
        os.makedirs(self.datasets_dirpath, exist_ok=True)
        os.makedirs(self.results_dirpath, exist_ok=True)
        os.makedirs(self.leaderboard_configs_dirpath, exist_ok=True)
        os.makedirs(self.leaderboard_csvs_dirpath, exist_ok=True)
        os.makedirs(self.summary_metrics_dirpath, exist_ok=True)

    def can_apply_summary_updates(self, cur_epoch):
        if self.last_summary_metric_update + self.summary_metric_update_timeframe <= cur_epoch:
            self.last_summary_metric_update = cur_epoch
            self.save_json()
            return True
        return False

    def __str__(self):
        msg = 'TestHarnessConfig: (\n'
        for key, value in self.__dict__.items():
            msg += '\t{} = {}\n'.format(key, value)
        msg += ')'
        return msg

    def save_json(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.test_harness_dirpath, TestHarnessConfig.CONFIG_FILENAME)
        json_io.write(filepath, self)

    @staticmethod
    def load_json(filepath) -> 'TestHarnessConfig':
        return json_io.read(filepath)


def init_cmd(args):
    test_harness_config = TestHarnessConfig(test_harness_dirpath=args.test_harness_dirpath, token_pickle_filepath=args.token_pickle_filepath,
                                      init=args.init, control_slurm_queue_name=args.control_slurm_queue_name)
    test_harness_config.save_json()
    print('Created: {}'.format(test_harness_config))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Creates test harness config')
    parser.set_defaults(func=lambda args: parser.print_help())

    subparser = parser.add_subparsers(dest='cmd', required=True)


    init_parser = subparser.add_parser('init')
    init_parser.add_argument('--test-harness-dirpath', type=str,
                        help='The main test harness directory path',
                        required=True)
    init_parser.add_argument('--token-pickle-filepath', type=str, help='The token pickle filepath', required=True)
    init_parser.add_argument('--control-slurm-queue-name', type=str, help='The name of the slurm queue used for control', default='control')
    init_parser.add_argument('--init', action='store_true')
    init_parser.set_defaults(func=init_cmd)

    args = parser.parse_args()

    args.func(args)
