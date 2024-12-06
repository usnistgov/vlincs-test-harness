from leaderboards.actor import ActorManager, Actor
from leaderboards.leaderboard import *
from leaderboards.mail_io import VLINCSMail
from leaderboards import json_io
from leaderboards import hash_utils
from leaderboards.submission_io_utils import init_submission_io
from leaderboards.tasks import Task
import time
import logging
import traceback
import os
import subprocess


def main(test_harness_config: TestHarnessConfig, leaderboard: Leaderboard, data_split_name: str,
         vm_name: str, team_name: str, team_email: str, submission_filepath: str, results_dirpath: str, job_id: str, submission_io_str: str):

    actor_manager = ActorManager.load_json(test_harness_config)
    actor = actor_manager.get_from_name(team_name)

    logging.info('**************************************************')
    logging.info('Executing Container within VM for team: {} within VM: {}'.format(team_name, vm_name))
    logging.info('**************************************************')

    errors = ''
    info_dict = {}
    submission_dirpath = os.path.dirname(submission_filepath)

    if not os.path.exists(submission_dirpath):
        os.makedirs(submission_dirpath)

    if not os.path.exists(results_dirpath):
        os.makedirs(results_dirpath)

    submission_metadata_filepath = os.path.join(submission_dirpath, team_name + '.metadata.json')
    # error_filepath = os.path.join(results_dirpath, 'errors.txt')
    info_file = os.path.join(results_dirpath, 'info.json')
    try:
        vm_ip = test_harness_config.vms[vm_name]
    except:
        msg = 'VM "{}" ended up in the wrong SLURM queue.\n{}'.format(vm_name, traceback.format_exc())
        errors += ":VM:"
        logging.error(msg)
        logging.error('config: "{}"'.format(test_harness_config))
        VLINCSMail().send(to='vlincs@nist.gov', subject='VM "{}" In Wrong SLURM Queue'.format(vm_name), message=msg)
        raise

    task : Task = leaderboard.task
    dataset = leaderboard.get_dataset(data_split_name)
    train_dataset = None
    if leaderboard.has_dataset(leaderboard.get_training_dataset_name()):
        train_dataset = leaderboard.get_dataset(leaderboard.get_training_dataset_name())

    # Step 1) Download the submission (if it does not exist)
    if not os.path.exists(submission_filepath):
        logging.info('Downloading file for "{}" from "{}"'.format(team_name, team_email))
        submission_dir = os.path.dirname(submission_filepath)
        submission_name = os.path.basename(submission_filepath)


        submission_io = init_submission_io(args.submission_io, test_harness_config)

        submission_file = submission_io.submission_download(team_email, submission_dir, submission_metadata_filepath, leaderboard.name, data_split_name)

        if submission_name != submission_file.name:
            logging.info('Name of file has changed since launching submission')
            submission_name = submission_file.name
            submission_filepath = os.path.join(submission_dir, submission_name)

        # Scan download
        if test_harness_config.scanner_script_filepath is not None:
            logging.info('Scanning submission {}'.format(submission_filepath))
            try:
                subprocess.run([test_harness_config.scanner_script_filepath, submission_filepath], check=True)
            except subprocess.CalledProcessError as e:
                logging.error('Scan FAILED')
                logging.error(str(e))
                VLINCSMail().send(to='vlincs@nist.gov', subject='Scan failed for submission',
                                  message='Scan failed for {}\n\n{}'.format(submission_filepath, str(e)))
                os.remove(submission_filepath)
                raise
            else:
                logging.info('Scan PASSED')

    # Step 2) Compute hash of container (if it does not exist)
    hash_utils.compute_hash(submission_filepath)

    # Step 3) Run basic VM task checks: check_gpu
    errors += task.run_basic_checks(vm_ip, vm_name)

    # Step 4) Check task parameters in container (files and directories, schema checker)
    submission_errors = task.run_submission_checks(submission_filepath, dataset)
    errors += submission_errors

    # Step 4a) Copy in environment to VM
    errors += task.copy_in_env(vm_ip, vm_name, test_harness_config)

    # Step 5) Run basic VM cleanups (scratch)
    errors += task.cleanup_vm(vm_ip, vm_name)

    # Add some delays
    time.sleep(2)

    # Step 6) Copy in and update permissions task data/scripts (submission, eval_scripts, training dataset, model dataset, other per-task data (tokenizers), source_data)
    errors += task.copy_in_task_data(vm_ip, vm_name, submission_filepath, dataset, train_dataset, leaderboard.excluded_files)

    # Add some delays
    time.sleep(2)

    # Step 7) Execute submission and check errors
    errors += task.execute_submission(vm_ip, vm_name, test_harness_config.evaluate_python_env, submission_filepath, dataset, train_dataset, leaderboard.excluded_files, info_dict, results_dirpath)

    # Add some delays
    time.sleep(2)

    # Step 8) Copy out results
    errors += task.copy_out_results(vm_ip, vm_name, results_dirpath)

    # Add some delays
    time.sleep(2)

    # Step 9) Process submissions within task
    errors += task.process_metrics(results_dirpath, dataset, leaderboard.submission_metrics, team_name, leaderboard.name)

    # Step 10) Re-run basic VM cleanups
    # TODO add back in
    # errors += task.cleanup_vm(vm_ip, vm_name, custom_remote_home, custom_remote_scratch_with_job_id)

    logging.info('**************************************************')
    logging.info('Container Execution Complete for team: {}'.format(team_name))
    logging.info('**************************************************')

    # Step 10) Update info dictionary (execution, errors)
    info_dict['errors'] = errors

    # Build per model execution time dictionary
    model_execution_time_dict = dict()
    for model_execution_time_file_name in os.listdir(results_dirpath):
        if not model_execution_time_file_name.endswith('-walltime.txt'):
            continue

        model_name = model_execution_time_file_name.split('-walltime')[0]
        model_execution_time_filepath = os.path.join(results_dirpath, model_execution_time_file_name)

        if not os.path.exists(model_execution_time_filepath):
            continue

        try:
            with open(model_execution_time_filepath, 'r') as execution_time_fh:
                line = execution_time_fh.readline().strip()
                while line:
                    if line.startswith('execution_time'):
                        toks = line.split(' ')
                        model_execution_time_dict[model_name] = float(toks[1])
                    else:
                        model_execution_time_dict[model_name] = float(line)
                    line = execution_time_fh.readline().strip()

        except:
            pass  # Do nothing if file fails to parse
        # delete the walltime file to avoid cluttering the output folder
        os.remove(model_execution_time_filepath)

    info_dict['model_execution_runtimes'] = model_execution_time_dict

    json_io.write(info_file, info_dict)


if __name__ == '__main__':
    import argparse

    # logs written to stdout are captured by slurm
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)d] %(message)s")

    parser = argparse.ArgumentParser(description='Starts/Stops VMs')
    parser.add_argument('--team-name', type=str,
                        help='The team name',
                        required=True)
    parser.add_argument('--team-email', type=str,
                        help='The team email',
                        required=True)
    parser.add_argument('--container-filepath', type=str,
                        help='The filepath to download the container.',
                        required=True)
    parser.add_argument('--result-dirpath', type=str,
                        help='The result directory for the team',
                        required=True)
    parser.add_argument('--test-harness-config-filepath', type=str,
                        help='The JSON file that describes the test harness',
                        default='config.json')
    parser.add_argument('--leaderboard-name', type=str,
                        help='The name of the leaderboards')
    parser.add_argument('--data-split-name', type=str, help='The name of the data split that we are executing on.')
    parser.add_argument('--vm-name', type=str,
                        help='The name of the vm.',
                        required=True)
    parser.add_argument('--job-id', type=str, help='The slurm job ID', default=None)
    parser.add_argument('--submission-io', type=str, choices=['g_drive'], default='g_drive', required=False)

    args = parser.parse_args()

    test_harness_config = TestHarnessConfig.load_json(args.test_harness_config_filepath)

    leaderboard = Leaderboard.load_json(test_harness_config, args.leaderboard_name)

    main(test_harness_config, leaderboard, args.data_split_name, args.vm_name, args.team_name, args.team_email,
         args.container_filepath, args.results_dirpath, args.job_id, args.submission_io)



