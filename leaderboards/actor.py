# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
import math
import os
from typing import KeysView, ValuesView
import json_io
import slurm
import time_utils
from leaderboard import *
from test_harness_config import TestHarnessConfig
from airium import Airium
import uuid



class Actor(object):
    VALID_TYPES = ['public', 'performer']
    def __init__(self, test_harness_config: TestHarnessConfig, email: str, name: str, poc_email: str, type: str, reset: bool = True):
        self.uuid = str(uuid.uuid1())
        self.email = email
        self.name = name
        self.poc_email = poc_email
        self.prior_emails = []
        self.prior_names = []
        self.type = type

        self.last_submission_epochs = {}
        self.last_file_epochs = {}

        self.general_file_status = 'None'

        self.job_statuses = {}
        self.file_statuses = {}

        if reset:
            for leaderboard_name in test_harness_config.active_leaderboard_names:
                leaderboard = Leaderboard.load_json(test_harness_config, leaderboard_name)
                for dataset_split_name in leaderboard.get_submission_data_split_names():
                    self.reset_leaderboard_submission(leaderboard_name, dataset_split_name)

        self.highlight_old_submissions = False
        self.disabled = False

    # Updates the job status for the actor, only updating the job status for the check values
    # if check value is none, then it will update all job statuses to the value
    def update_all_job_status(self, value, check_value=None):
        for key, old_value in self.job_statuses.items():
            if check_value is None:
                self.job_statuses[key] = value
            elif check_value == old_value:
                self.job_statuses[key] = value

    def has_job_status(self, check_value):
        for value in self.job_statuses.values():
            if check_value == value:
                return True
        return False

    def get_last_file_epoch(self, leaderboard_name, data_split_name):
        return self.last_file_epochs[self.get_leaderboard_key(leaderboard_name, data_split_name)]

    def get_last_submission_epoch(self, leaderboard_name, data_split_name):
        return self.last_submission_epochs[self.get_leaderboard_key(leaderboard_name, data_split_name)]

    def update_job_status(self, leaderboard_name, data_split_name, value):
        self.job_statuses[self.get_leaderboard_key(leaderboard_name, data_split_name)] = value

    def update_file_status(self, leaderboard_name, data_split_name, value):
        self.file_statuses[self.get_leaderboard_key(leaderboard_name, data_split_name)] = value

    def update_last_submission_epoch(self, leaderboard_name, data_split_name, value):
        self.last_submission_epochs[self.get_leaderboard_key(leaderboard_name, data_split_name)] = value

    def update_last_file_epoch(self, leaderboard_name, data_split_name, value):
        self.last_file_epochs[self.get_leaderboard_key(leaderboard_name, data_split_name)] = value

    def get_leaderboard_key(self, leaderboard_name, data_split_name):
        key = '{}_{}'.format(leaderboard_name, data_split_name)
        if key not in self.last_submission_epochs.keys() or key not in self.last_file_epochs.keys() or key not in self.job_statuses.keys() or key not in self.file_statuses.keys():
            self.last_submission_epochs[key] = 0
            self.last_file_epochs[key] = 0
            self.job_statuses[key] = 'None'
            self.file_statuses[key] = 'None'
        return key


    def _has_leaderboard_metadata(self, leaderboard_name, data_split_name):
        leaderboard_key = self.get_leaderboard_key(leaderboard_name, data_split_name)
        return leaderboard_key in self.last_file_epochs.keys() and leaderboard_key in self.last_submission_epochs.keys() \
               and leaderboard_key in self.job_statuses.keys() and leaderboard_key in self.file_statuses.keys()

    def reset_leaderboard_submission(self, leaderboard_name, data_split_name):
        print('Resetting {} for leaderboard: {} and data split {}'.format(self.email, leaderboard_name, data_split_name))
        leaderboard_key = self.get_leaderboard_key(leaderboard_name, data_split_name)
        self.last_submission_epochs[leaderboard_key] = 0
        self.last_file_epochs[leaderboard_key] = 0
        self.job_statuses[leaderboard_key] = 'None'
        self.file_statuses[leaderboard_key] = 'None'

    def reset_leaderboard_time_window(self, leaderboard_name, data_split_name):
        self.update_last_submission_epoch(leaderboard_name, data_split_name, 0)

    def __str__(self):
        msg = 'Actor: (\n'
        for key, value in self.__dict__.items():
            msg += '\t{} = {}\n'.format(key, value)
        msg += ')'
        return msg

    def save_json(self, filepath: str) -> None:
        json_io.write(filepath, self)

    @staticmethod
    def load_json(filepath: str) -> 'Actor':
        return json_io.read(filepath)

    def in_queue(self, queue_name) -> bool:
        stdout, stderr = slurm.squeue(self.name, queue_name)
        stdout = str(stdout).split("\\n")
        return len(stdout) > 2

    def is_disabled(self):
        return self.disabled

    def can_submit_time_window(self, leaderboard_name, dataset_split_name, execute_window_seconds, cur_epoch) -> bool:
        # Check if the actor is allowed to execute again or not
        last_execution_epoch = self.get_last_submission_epoch(leaderboard_name, dataset_split_name)
        if last_execution_epoch + execute_window_seconds <= cur_epoch:
            return True
        return False

    def get_jobs_table_row(self, a: Airium, leaderboard_name: str, leaderboard_highlight_old_submissions: bool, data_split_name: str, execute_window: int , current_epoch: int, job_color_key: dict):
        leaderboard_key = self.get_leaderboard_key(leaderboard_name, data_split_name)

        # Check if this is the first time we've encountered this leaderboard
        if not self._has_leaderboard_metadata(leaderboard_name, data_split_name):
            self.reset_leaderboard_submission(leaderboard_name, data_split_name)

        remaining_time = 0
        last_execution_epoch = self.last_submission_epochs[leaderboard_key]
        if last_execution_epoch + execute_window > current_epoch:
            remaining_time = (last_execution_epoch + execute_window) - current_epoch

        days, hours, minutes, seconds = time_utils.convert_seconds_to_dhms(remaining_time)
        time_str = "{} d, {} h, {} m, {} s".format(days, hours, minutes, seconds)

        if self.last_file_epochs[leaderboard_key] == 0:
            last_file_timestamp = "None"
        else:
            last_file_timestamp = time_utils.convert_epoch_to_iso(self.last_file_epochs[leaderboard_key])
        if self.last_submission_epochs[leaderboard_key] == 0:
            last_submission_timestamp = ""
        else:
            last_submission_timestamp = time_utils.convert_epoch_to_iso(self.last_submission_epochs[leaderboard_key])

        color_key_times = sorted([float(i) for i in job_color_key.keys()])
        color_class = ''

        if self.highlight_old_submissions and leaderboard_highlight_old_submissions:
            # Find the color for the row
            for color_key_time in color_key_times:
                if math.isinf(color_key_time):
                    color_class = job_color_key['inf']
                    break
                elif last_execution_epoch + int(color_key_time) > current_epoch:
                    color_class = job_color_key[str(int(color_key_time))]
                    break

        with a.tr():
            a.td(_t=self.name)
            a.td(klass=color_class,_t=last_submission_timestamp)
            a.td(_t=self.type)
            a.td(_t=self.job_statuses[leaderboard_key])
            a.td(_t=self.file_statuses[leaderboard_key])
            a.td(_t=self.general_file_status)
            a.td(_t=last_file_timestamp)
            a.td(_t=time_str)


class ActorManager(object):
    def __init__(self):
        self.actors = dict()

    def __str__(self):
        msg = "Actors: \n"
        for actor_id, actor in self.actors.items():
            msg = msg + "  " + actor.__str__() + "\n"
        return msg

    def get_keys(self) -> KeysView:
        return self.actors.keys()

    def get_actors(self) -> ValuesView:
        return self.actors.values()

    def add_actor(self, test_harness_config: TestHarnessConfig, email: str, name: str, poc_email: str, actor_type: str) -> None:
        for actor in self.actors.values():
            if email == actor.email:
                raise RuntimeError("Actor already exists in ActorManager: {}".format(email))
            if name == actor.name:
                raise RuntimeError("Actor Name already exists in ActorManager: {}".format(name))
        created_actor = Actor(test_harness_config, email, name, poc_email, actor_type)
        self.actors[str(created_actor.uuid)] = created_actor
        print('Created: {}'.format(created_actor))

    def remove_actor(self, email) -> None:
        actor = self.get(email)
        del self.actors[actor.uuid]
        print('Removed {} from actor manager'.format(email))

    def get(self, email) -> Actor:
        actors = []
        for actor in self.actors.values():
            if actor.email == email:
                actors.append(actor)

        if len(actors) == 0:
            raise RuntimeError('Unable to find email in ActorManager: {}'.format(email))

        if len(actors) > 1:
            raise RuntimeError('Multiple actors share the same email in ActorManager: {}'.format(email))

        return actors[0]

    def get_from_name(self, actor_name) -> Actor:
        actors = []
        for actor in self.actors.values():
            if actor.name == actor_name:
                actors.append(actor)

        if len(actors) == 0:
            for actor in self.actors.values():
                for prior_name in actor.prior_names:
                    if prior_name == actor_name:
                        actors.append(actor)

        if len(actors) == 0:
            raise RuntimeError('Unable to find actor name in ActorManager: {}'.format(actor_name))

        if len(actors) > 1:
            raise RuntimeError('Multiple actors share the sasme name in ActorManager: {}'.format(actor_name))

        return actors[0]

    def get_from_uuid(self, uuid) -> Actor:
        if str(uuid) in self.actors.keys():
            return self.actors[str(uuid)]
        else:
            raise RuntimeError('Invalid uuid key {}, not found in actor manager'.format(uuid))

    def save_json(self, test_harness_config: TestHarnessConfig) -> None:
        json_io.write(test_harness_config.actors_filepath, self)

    @staticmethod
    def init_file(test_harness_config: TestHarnessConfig) -> None:
        # Create the json file if it does not exist already
        if not os.path.exists(test_harness_config.actors_filepath):
            actor_dict = ActorManager()
            actor_dict.save_json(test_harness_config)

    @staticmethod
    def load_json(test_harness_config: TestHarnessConfig) -> 'ActorManager':
        # make sure the file exists
        if not os.path.exists(test_harness_config.actors_filepath):
            ActorManager.init_file(test_harness_config)

        return json_io.read(test_harness_config.actors_filepath)

    def write_jobs_table(self, output_dirpath, leaderboard_name, leaderboard_highlight_old_submissions, dataset_split_name, execute_window, cur_epoch, job_color_key):
        jobs_filename = 'jobs-{}-{}.html'.format(leaderboard_name, dataset_split_name)
        jobs_filepath = os.path.join(output_dirpath, leaderboard_name, jobs_filename)
        a = Airium()

        with a.div(klass='card-body card-body-cascade pb-0'):
            a.h2(klass='pb-q card-title', _t='Teams/Jobs')
            with a.div(klass='table-responsive'):
                with a.table(id='{}-{}-jobs'.format(leaderboard_name, dataset_split_name), klass='table table-striped table-bordered table-sm'):
                    with a.thead():
                        with a.tr():
                            a.th(klass='th-sm', _t='Team')
                            a.th(klass='th-sm', _t='Submission Timestamp')
                            a.th(klass='th-sm', _t='Type')
                            a.th(klass='th-sm', _t='Job Status')
                            a.th(klass='th-sm', _t='File Status')
                            a.th(klass='th-sm', _t='General Status')
                            a.th(klass='th-sm', _t='File Timestamp')
                            a.th(klass='th-sm', _t='Time until next execution')
                    with a.tbody():
                        for actor in self.actors.values():
                            actor.get_jobs_table_row(a, leaderboard_name, leaderboard_highlight_old_submissions, dataset_split_name, execute_window, cur_epoch, job_color_key)

        with open(jobs_filepath, 'w') as f:
            f.write(str(a))

        return jobs_filepath

    def convert_to_csv(self, output_file):
        write_header = True

        with open(output_file, 'w') as file:
            for actor in self.actors.values():
                if write_header:
                    for key in actor.__dict__.keys():
                        file.write(key + ',')
                    file.write('\n')
                    write_header = False

                for key in actor.__dict__.keys():
                    file.write(str(actor.__dict__[key]) + ',')
                file.write('\n')

def add_actor_helper(test_harness_config: TestHarnessConfig, team_name: str, email: str, poc_email: str, type: str):
    actor_manager = ActorManager.load_json(test_harness_config)
    try:
        team_name = team_name.encode('ascii')
    except:
        raise RuntimeError('Team name must be ASCII only')

    team_name = team_name.decode()
    team_name = str(team_name)
    invalid_chars = [" ", "/", ">", "<", "|", ":", "&", ",", ";", "?", "\\", "*"]
    for char in invalid_chars:
        if char in team_name:
            raise RuntimeError('team_name cannot have invalid characters: {}'.format(invalid_chars))

    actor_manager.add_actor(test_harness_config, email, team_name, poc_email, type)
    actor_manager.save_json(test_harness_config)

def add_actor(args):
    test_harness_config = TestHarnessConfig.load_json(args.test_harness_config_filepath)
    team_name = args.name
    email = args.email
    poc_email = args.poc_email
    type = args.type
    add_actor_helper(test_harness_config, team_name, email, poc_email, type)

def remove_actor(args):
    test_harness_config = TestHarnessConfig.load_json(args.test_harness_config_filepath)
    actor_manager = ActorManager.load_json(test_harness_config)
    actor_manager.remove_actor(args.email)
    actor_manager.save_json(test_harness_config)


def reset_actor(args):
    test_harness_config = TestHarnessConfig.load_json(args.test_harness_config_filepath)
    actor_manager = ActorManager.load_json(test_harness_config)

    data_splits = set()
    leaderboards = []

    email = args.email
    leaderboard_name = args.leaderboard
    data_split = args.data_split

    if leaderboard_name is not None:
        leaderboard = Leaderboard.load_json(test_harness_config, leaderboard_name)
        leaderboards.append(leaderboard)
        if data_split is not None:
            data_splits.add(data_split)
        else:
            data_splits.update(leaderboard.get_submission_data_split_names())
    else:
        for leaderboard_name in test_harness_config.active_leaderboard_names:
            leaderboard = Leaderboard.load_json(test_harness_config, leaderboard_name)
            leaderboards.append(leaderboard)
            data_splits.update(leaderboard.get_submission_data_split_names())

    actor = actor_manager.get(email)

    for leaderboard in leaderboards:
        for data_split_name in data_splits:
            if leaderboard.can_submit_to_dataset(data_split_name):
                actor.reset_leaderboard_submission(leaderboard.name, data_split_name)
            else:
                print('WARNING: Unable to submit to leaderboards {} for split {}, did not reset'.format(leaderboard.name, data_split_name))

    actor_manager.save_json(test_harness_config)


def actor_to_csv(args):
    test_harness_config = TestHarnessConfig.load_json(args.test_harness_config_filepath)
    actor_manager = ActorManager.load_json(test_harness_config)
    actor_manager.convert_to_csv(args.output_filepath)

def apply_fix_actor_manager(args):
    test_harness_config = TestHarnessConfig.load_json(args.test_harness_config_filepath)
    actor_manager = ActorManager.load_json(test_harness_config)

    fixed_actors = {}
    for actor in actor_manager.get_actors():
        actor.uuid = str(actor.uuid)
        fixed_actors[str(actor.uuid)] = actor

    actor_manager.actors = fixed_actors

    actor_manager.save_json(test_harness_config)

if __name__ == "__main__":
    import argparse

    uuid_test = uuid.uuid1()

    test_str = str(uuid_test)
    print(test_str)

    parser = argparse.ArgumentParser(description='Creates an actor, and adds it to the actors.json')
    parser.set_defaults(func=lambda args: parser.print_help())

    subparser = parser.add_subparsers(dest='cmd', required=True)

    add_actor_parser = subparser.add_parser('add-actor')
    add_actor_parser.add_argument('--test-harness-config-filepath', type=str, help='The filepath to the main test harness config', required=True)
    add_actor_parser.add_argument('--name', type=str, help='The name of the team to add', required=True)
    add_actor_parser.add_argument('--email', type=str, help='The submission email of the team to add', required=True)
    add_actor_parser.add_argument('--poc-email', type=str, help='The point of contact email of the team to add', required=True)
    add_actor_parser.add_argument('--type', type=str, choices=Actor.VALID_TYPES, help='The type of actor, displayed on the jobs table in the HTML', default='public')
    add_actor_parser.set_defaults(func=add_actor)

    remove_actor_parser = subparser.add_parser('remove-actor')
    remove_actor_parser.add_argument('--test-harness-config-filepath', type=str, help='The filepath to the main test harness config', required=True)
    remove_actor_parser.add_argument('--email', type=str, help='The email of the team to remove', required=True)
    remove_actor_parser.set_defaults(func=remove_actor)

    reset_actor_parser = subparser.add_parser('reset-actor')
    reset_actor_parser.add_argument('--test-harness-config-filepath', type=str, help='The filepath to the main test harness config', required=True)
    reset_actor_parser.add_argument('--email', type=str, help='The email of the team to reset', required=True)
    reset_actor_parser.add_argument('--leaderboard', type=str, help='The name of the leaderboard to reset, if used by itself will reset all data splits', default=None)
    reset_actor_parser.add_argument('--data-split', type=str, help='The data split name to reset associated with leaderboards. Will only reset that leaderboards and data split.', default=None)
    reset_actor_parser.set_defaults(func=reset_actor)

    to_csv_parser = subparser.add_parser('to-csv')
    to_csv_parser.add_argument('--test-harness-config-filepath', type=str, help='The filepath to the main test harness config', required=True)
    to_csv_parser.add_argument('--output-filepath', type=str, help='The output filepath for the csv', default='actors.csv')
    to_csv_parser.set_defaults(func=actor_to_csv)

    fix_actor_manager = subparser.add_parser('fix')
    fix_actor_manager.add_argument('--test-harness-config-filepath', type=str, help='The filepath to the main test harness config', required=True)
    fix_actor_manager.set_defaults(func=apply_fix_actor_manager)

    # TODO: Add update function to safely update various attributes of an actor
    # TODO: Add ability to rename the name of an actor

    args = parser.parse_args()

    args.func(args)