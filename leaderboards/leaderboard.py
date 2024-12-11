# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import collections
import copy
import datetime
import json
import logging
import os
import typing

import jsonpickle
import pandas as pd
import numpy as np
from airium import Airium
from typing import Union, Tuple, Dict, Any

from submission_io import SubmissionIO
from python_utils import update_object_values, get_value
from test_harness_config import TestHarnessConfig
import json_io
from dataset import DatasetManager, Dataset, VideoLINCSDataset
from metrics import *
from tasks import *
from summary_metrics import *
from results_manager import ResultsManager


class Leaderboard(object):
    LEADERBOARD_TYPES = ['vlincs']

    ALL_TASK_NAMES = {'take_home': TakeHomeTask}

    ALL_METRIC_NAMES = {'test': TestMetric}

    ALL_SUMMARY_METRIC_NAMES = {}

    GENERAL_SLURM_QUEUE_NAME = 'es'
    STS_SLURM_QUEUE_NAME = 'sts'
    INFO_FILENAME = 'info.json'
    METRIC_RESULTS_FILENAME = 'metric_results.json'

    SLURM_QUEUE_NAMES = [GENERAL_SLURM_QUEUE_NAME, STS_SLURM_QUEUE_NAME]

    TABLE_NAMES = ['results', 'all-results', 'jobs']

    DEFAULT_SCHEMA_SUMMARY_SUFFIX = 'schema_summary.csv'

    # This is a running list of unique dataset names to show tooltip text
    DATASET_DESCRIPTIONS = {
        'train': 'train: The train dataset that is distributed with each round.',
        'test': 'test: The test dataset that is sequestered/hidden, used for evaluation. Submissions here should be fully realized with complete schema and parameters.',
        'sts': 'sts: The sts dataset uses a subset of the train dataset, useful for debugging container submission.',
        'dev': 'dev: The dev dataset uses the test dataset, and should be used for in-development solutions. Schemas must be valid, but do not need to be complete. Results do not count towards the program.',
        'holdout': 'holdout: The holdout dataset that is sequestered/hidden, used for holdout evaluation.'
    }
    def __init__(self, name: str, task_name: str, test_harness_config: TestHarnessConfig):
        if '_' in name:
            raise RuntimeError('Invalid leaderboard name: {}, should not have any underscores "_"'.format(name))
        self.name = name
        self.task_name = task_name
        self.revision = 1
        self.check_for_missing_metrics = True

        self.submission_dirpath = os.path.join(test_harness_config.submission_dirpath, self.name)
        self.submissions_filepath = os.path.join(self.submission_dirpath, 'submissions.json')

        self.leaderboard_results_filepath = os.path.join(test_harness_config.leaderboard_results_dirpath, '{}_results.parquet'.format(self.name))

        self.highlight_old_submissions = False
        self.html_leaderboard_priority = 0
        self.html_data_split_name_priorities = {}
        self.html_table_sort_options = {}

        self.evaluation_metric_name = None
        self.summary_metrics = []
        self.submission_metrics: Dict[str, Metric] = {}

        self.excluded_files = []
        self.required_files = []

        self.summary_metadata_csv_filepath = os.path.join(test_harness_config.leaderboard_csvs_dirpath, '{}_METADATA.csv'.format(self.name))
        self.summary_results_csv_filepath = os.path.join(test_harness_config.leaderboard_csvs_dirpath, '{}_RESULTS.csv'.format(self.name))

        self.dataset_manager = DatasetManager()
        self.task = None

    ########################
    # Utility IO functions
    ########################
    def save_json(self, test_harness_config: TestHarnessConfig):
        filepath = os.path.join(test_harness_config.leaderboard_configs_dirpath, '{}_config.json'.format(self.name))
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        json_io.write(filepath, self)

    @staticmethod
    def load_json(test_harness_config: TestHarnessConfig, name: str) -> Union['Leaderboard', None]:
        leaderboard_config_filepath = os.path.join(test_harness_config.leaderboard_configs_dirpath, '{}_config.json'.format(name))

        if not os.path.exists(leaderboard_config_filepath):
            logging.error('Unable to find leaderboard config: {}'.format(leaderboard_config_filepath))
            return None

        leaderboard_config = json_io.read(leaderboard_config_filepath)
        assert leaderboard_config.task_name in Leaderboard.ALL_TASK_NAMES

        leaderboard_config.check_instance_data(test_harness_config)

        return leaderboard_config

    def get_summary_schema_csv_filepath(self, test_harness_config: TestHarnessConfig):
        return os.path.join(test_harness_config.summary_metrics_dirpath, '{}-schema-summary.csv'.format(self.name))

    def load_summary_results_csv_into_df(self):
        if os.path.exists(self.summary_results_csv_filepath):
            try:
                return pd.read_csv(self.summary_results_csv_filepath)
            except pd.errors.EmptyDataError:
                return None
        else:
            logging.warning('Unable to find summary results metadata_csv at location: {}, please generate it through the submission manager.'.format(self.summary_results_csv_filepath))
            return None

    def load_metadata_csv_into_df(self):
        if not os.path.exists(self.summary_metadata_csv_filepath):
            logging.warning('Unable to find summary metadata_csv at location: {}, generating CSV now.'.format(self.summary_metadata_csv_filepath))
            self.generate_metadata_csv()

        # If it still doesn't exist, then there might be no data to load
        if not os.path.exists(self.summary_metadata_csv_filepath):
            return None

        return pd.read_csv(self.summary_metadata_csv_filepath)

    def load_results_df(self, results_manager: ResultsManager):
        return results_manager.load_results(self.name, self.leaderboard_results_filepath, self.get_default_result_columns())

    def update_results_df(self, results_manager: ResultsManager, df):
        results_manager.update_results(self.name, df)

    ########################
    # Dataset utility functions
    ########################
    def get_default_prediction_result(self):
        return self.task.default_prediction_result

    def get_dataset(self, data_split_name):
        return self.dataset_manager.get(data_split_name)

    def has_dataset(self, data_split_name):
        return self.dataset_manager.has_dataset(data_split_name)

    def load_ground_truth(self, data_split_name) -> typing.OrderedDict[str, float]:
        raise NotImplementedError()

    def get_result_dirpath(self, data_split_name):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.results_dirpath

    def get_slurm_queue_name(self, data_split_name):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.slurm_queue_name

    def get_slurm_nice(self, data_split_name: str):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.slurm_nice

    def is_auto_delete_submission(self, data_split_name: str):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.auto_delete_submission

    def get_auto_execute_split_names(self, data_split_name: str):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.auto_execute_split_names

    def get_submission_window_time(self, data_split_name: str):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.submission_window_time_sec

    def can_submit_to_dataset(self, data_split_name: str):
        return self.dataset_manager.can_submit_to_dataset(data_split_name)

    def get_submission_data_split_names(self):
        return self.dataset_manager.get_submission_dataset_split_names()

    def get_html_data_split_names(self):
        return self.html_data_split_name_priorities.keys()

    def get_all_data_split_names(self):
        return self.dataset_manager.datasets.keys()

    def get_timeout_time_sec(self, data_split_name):
        dataset = self.dataset_manager.get(data_split_name)
        return dataset.timeout_time_sec

    def get_dataset_size(self, data_split_name):
        dataset = self.dataset_manager.get(data_split_name)
        return len(dataset)

    def check_instance_data(self, test_harness_config: TestHarnessConfig):
        has_updated = False
        has_updated = has_updated or self.task.check_instance_params(test_harness_config)

        if not hasattr(self, 'check_for_missing_metrics'):
            has_updated = True
            self.check_for_missing_metrics = True

        if not hasattr(self, 'revision'):
            has_updated = True
            self.revision = 1

        if has_updated:
            self.save_json(test_harness_config)

    def add_dataset(self, test_harness_config: TestHarnessConfig,
                    split_name: str,
                    can_submit: bool,
                    slurm_queue_name: str,
                    slurm_nice: int,
                    generate_metadata_csv: bool=False,
                    on_html: bool=False):
        raise NotImplementedError()


    def initialize_directories(self):
        os.makedirs(self.submission_dirpath, exist_ok=True)
        self.dataset_manager.initialize_directories()

    def get_task(self) -> Task:
        return self.task

    def add_metric(self, metric: Metric):
        self.submission_metrics[metric.get_name()] = metric

    def remove_metric(self, metric_name: str):
        if metric_name not in self.submission_metrics:
            logging.info('Failed to remove {}, it does not exist'.format(metric_name))
        else:
            del self.submission_metrics[metric_name]

    def __str__(self):
        msg = 'Leaderboard: (\n'
        for key, value in self.__dict__.items():
            msg += '\t{} = {}\n'.format(key, value)
        msg += ')'
        return msg

    def write_html_leaderboard(self, is_accepting_submissions: bool, html_output_dirpath: str, is_first: bool, is_archived: bool):

        leaderboard_filename = '{}-leaderboard.html'.format(self.name)
        leaderboard_dirpath = os.path.join(html_output_dirpath, self.name)
        leaderboard_filepath = os.path.join(leaderboard_dirpath, leaderboard_filename)

        if not os.path.exists(leaderboard_dirpath):
            os.makedirs(leaderboard_dirpath, exist_ok=True)

        active_show = ''
        if is_first:
            active_show = 'active show'

        html_data_split_names = sorted(self.html_data_split_name_priorities, key=self.html_data_split_name_priorities.get, reverse=True)


        a = Airium()
        with a.div(klass='tab-pane fade {}'.format(active_show), id='{}'.format(self.name), role='tabpanel', **{'aria-labelledby' : 'tab-{}'.format(self.name)}):
            a('{{% include {}/about-{}.html %}}'.format(self.name, self.name))

            with a.div(klass='card-body card-body-cascade'):
                with a.p(klass='card-text text-left'):
                    for data_split in html_data_split_names:
                        if data_split in Leaderboard.DATASET_DESCRIPTIONS:
                            a('{}<br>'.format(Leaderboard.DATASET_DESCRIPTIONS[data_split]))

            with a.ul(klass='nav nav-pills', id='{}-tabs'.format(self.name), role='tablist'):
                with a.li(klass='nav-item'):
                    for data_split in html_data_split_names:
                        if data_split == 'test':
                            a.a(klass='nav-link waves-light active show', id='tab-{}-{}'.format(self.name, data_split), href='#{}-{}'.format(self.name, data_split), **{'data-toggle': 'tab', 'aria-controls': '{}-{}'.format(self.name, data_split), 'aria-selected': 'true'}, _t=data_split)
                        else:
                            a.a(klass='nav-link waves-light', id='tab-{}-{}'.format(self.name, data_split), href='#{}-{}'.format(self.name, data_split), **{'data-toggle': 'tab', 'aria-controls': '{}-{}'.format(self.name, data_split), 'aria-selected': 'false'}, _t=data_split)

            with a.div(klass='tab-content card'):
                for data_split in html_data_split_names:
                    if not self.has_dataset(data_split):
                        continue

                    if data_split == 'test':
                        active_show = 'active show'
                    else:
                        active_show = ''
                    with a.div(klass='tab-pane fade {}'.format(active_show), id='{}-{}'.format(self.name, data_split), role='tabpanel', **{'aria-labelledby': 'tab-{}-{}'.format(self.name, data_split)}):
                        with a.div(klass='card-body card-body-cascade'):
                            dataset = self.get_dataset(data_split)
                            # TODO: Might need to update submission filename format
                            required_format = 'Required filename format: "{}_{}_&lt;Submission Name&gt;.simg"'.format(self.name, data_split)
                            accepting_submissions_info = 'Accepting submissions: {}'.format(dataset.can_submit and not is_archived and is_accepting_submissions)
                            model_info = 'Dataset size in {}, {}: {}'.format(self.name, data_split, self.get_dataset_size(data_split))
                            time_info = 'Execution timeout (hh:mm:ss): {}'.format(str(datetime.timedelta(seconds=self.get_timeout_time_sec(data_split))))

                            if is_archived:
                                a.p(klass='card-text text-left', _t='{}<br>{}'.format(accepting_submissions_info, model_info))
                            else:
                                a.p(klass='card-text text-left', _t='{}<br>{}<br>{}<br>{}'.format(required_format, accepting_submissions_info, model_info, time_info))

                        if not is_archived:
                            a('{{% include {}/jobs-{}-{}.html %}}'.format(self.name, self.name, data_split))

                        a('{{% include {}/results-unique-{}-{}.html %}}'.format(self.name, self.name, data_split))
                        a('{{% include {}/results-{}-{}.html %}}'.format(self.name, self.name, data_split))

        with open(leaderboard_filepath, 'w') as f:
            f.write(str(a))

        return leaderboard_filepath

    def initialize_html_options(self, split_name: str, on_html: bool):
        if on_html:
            self.html_data_split_name_priorities[split_name] = 0
        for table_name in Leaderboard.TABLE_NAMES:
            key = '{}-{}-{}'.format(self.name, split_name, table_name)
            if table_name == 'jobs':
                self.html_table_sort_options[key] = {'column': 'Submission Timestamp', 'order': 'desc',
                                                     'split_name': split_name}
            else:
                if split_name == 'sts':
                    self.html_table_sort_options[key] = {'column': 'Submission Timestamp', 'order': 'desc',
                                                         'split_name': split_name}
                else:
                    if self.dataset_manager.has_dataset(split_name) and self.evaluation_metric_name is not None:
                        sort_order = 'desc'

                        evaluation_metric_name = self.evaluation_metric_name
                        evaluation_metric_sub_name = None
                        if '::' in evaluation_metric_name:
                            eval_metric_split = evaluation_metric_name.split(':')
                            evaluation_metric_name = eval_metric_split[0]
                            evaluation_metric_sub_name = eval_metric_split[1]

                        if evaluation_metric_name in self.submission_metrics:
                            sort_order = self.submission_metrics[evaluation_metric_name].get_sort_order()
                        self.html_table_sort_options[key] = {'column': evaluation_metric_sub_name,
                                                             'order': sort_order, 'split_name': split_name}
                    else:
                        self.html_table_sort_options[key] = {'column': 'Submission Timestamp', 'order': 'asc',
                                                             'split_name': split_name}

    def generate_metadata_csv(self, overwrite_csv: bool = True):
        all_df_list = []

        if os.path.exists(self.summary_metadata_csv_filepath) and not overwrite_csv:
            logging.warning('Skipping building round metadata: {} already exists and overwrite is disabled.'.format(self.summary_metadata_csv_filepath))
            return

        for split_name in self.get_all_data_split_names():
            dataset = self.get_dataset(split_name)
            dataset_dirpath = dataset.dataset_dirpath

            metadata_filepath = os.path.join(dataset_dirpath, Dataset.METADATA_NAME)

            if not os.path.exists(metadata_filepath):
                logging.warning('Skipping {}, it does not contain the metadata file: {}'.format(dir, metadata_filepath))
                continue

            df = pd.read_csv(metadata_filepath)

            # Add column for data_split
            new_df = df.assign(data_split=split_name)

            # Add column for ground_truth
            new_df = new_df.assign(ground_truth='NaN')

            models_dir = os.path.join(dataset_dirpath, 'models')

            if os.path.exists(models_dir):
                # Add ground truth values into data
                for model_name in os.listdir(models_dir):
                    model_dirpath = os.path.join(models_dir, model_name)

                    # Skip model_name that is not a directory
                    if not os.path.isdir(model_dirpath):
                        continue

                    ground_truth_filepath = os.path.join(models_dir, model_name, Dataset.GROUND_TRUTH_NAME)
                    if not os.path.exists(ground_truth_filepath):
                        logging.warning('WARNING, ground truth file does not exist: {}'.format(ground_truth_filepath))
                        continue

                    with open(ground_truth_filepath, 'r') as f:
                        data = float(f.read())
                        new_df.loc[new_df['model_name'] == model_name, 'ground_truth'] = data
            else:
                logging.warning('{} does not exist'.format(models_dir))

            all_df_list.append(new_df)

        if len(all_df_list) == 0:
            logging.warning('Skipping {}, it does not contain any datasets for metadata'.format(self.summary_metadata_csv_filepath))
            return

        all_df = pd.concat(all_df_list)

        # Rearrange columns slightly
        columns = all_df.columns.tolist()
        # TODO: Update column orders
        column_order = ['model_name', 'ground_truth', 'data_split']
        remove_columns = ['converged', 'nonconverged_reason', 'Unnamed: 0']

        # Remove columns
        for column_name in remove_columns:
            if column_name in columns:
                columns.remove(column_name)

        # Reorder columns
        index = 0
        for column_name in column_order:
            if column_name in columns:
                columns.remove(column_name)
                columns.insert(index, column_name)
                index += 1

        all_df = all_df[columns]

        all_df.to_csv(self.summary_metadata_csv_filepath, index=False)
        logging.info('Finished writing round metadata to {}'.format(self.summary_metadata_csv_filepath))

    def get_default_result_columns(self):
        return ['submission_timestamp', 'actor_name', 'actor_UUID', 'data_split']

    def update_results_csv(self, result_df: pd.DataFrame, results_manager: ResultsManager, submission_epoch: int, data_split: str, actor_name: str, actor_uuid: str):
        raise NotImplementedError()

    def get_valid_metric(self, metric_name):
        raise NotImplementedError()

    def get_valid_summary_metric(self, metric_name):
        raise NotImplementedError()

    def get_training_dataset_name(self):
        raise NotImplementedError()

    def process_metrics(self, submission_io: SubmissionIO, results_manager: ResultsManager, data_split_name: str, execution_results_dirpath: str, actor_name: str, actor_uuid: str, submission_epoch_str: str, processed_metrics: list, skip_upload_existing: bool, pre_processed_metrics_dict: dict):
        raise NotImplementedError()

class VideoLINCSLeaderboard(Leaderboard):
    GENERAL_SLURM_QUEUE_NAME = 'take-home'
    DEFAULT_METRICS = [TestMetric]
    DEFAULT_EVALUATION_METRIC_NAME = 'TestMetric'

    DEFAULT_EXCLUDED_FILES = []
    DEFAULT_REQUIRED_FILES = []

    DEFAULT_DATASET_SPLIT_NAMES = ['test', 'sts']
    DEFAULT_SUBMISSION_DATASET_SPLIT_NAMES = ['test', 'sts']

    VALID_TASK_NAMES = {'take_home': TakeHomeTask}

    VALID_METRIC_NAMES = {
        'TestMetric': TestMetric
    }

    def __init__(self, name: str, task_name: str, test_harness_config: TestHarnessConfig, add_default_data_split: bool):
        super().__init__(name, task_name, test_harness_config)

        if self.task_name not in VideoLINCSLeaderboard.VALID_TASK_NAMES:
            raise RuntimeError('Invalid task name: {}'.format(self.task_name))

        self.task = VideoLINCSLeaderboard.VALID_TASK_NAMES[self.task_name](test_harness_config, self.name)
        self.evaluation_metric_name = TestMetric().get_name()

        self.excluded_files.extend(VideoLINCSLeaderboard.DEFAULT_EXCLUDED_FILES)
        self.required_files.extend(VideoLINCSLeaderboard.DEFAULT_REQUIRED_FILES)

        for metric in VideoLINCSLeaderboard.DEFAULT_METRICS:
            new_metric = metric()
            self.submission_metrics[new_metric.get_name()] = new_metric

        if add_default_data_split:
            for split_name in VideoLINCSLeaderboard.DEFAULT_DATASET_SPLIT_NAMES:
                if split_name in VideoLINCSLeaderboard.DEFAULT_SUBMISSION_DATASET_SPLIT_NAMES:
                    can_submit = True
                    on_html = True
                else:
                    can_submit = False
                    on_html = False

                auto_delete_submission = False
                slurm_nice = 10
                if split_name == 'sts':
                    slurm_queue_name = Leaderboard.STS_SLURM_QUEUE_NAME
                    slurm_nice = 0
                else:
                    if task_name == 'take_home':
                        slurm_queue_name = 'take-home'
                    else:
                        slurm_queue_name = Leaderboard.GENERAL_SLURM_QUEUE_NAME

                self.add_dataset(test_harness_config, split_name, can_submit, slurm_queue_name, slurm_nice, on_html=on_html)

        self.initialize_directories()
        # TODO: Decide on metadata CSV
        #self.generate_metadata_csv()

    def process_metrics(self, submission_io: SubmissionIO, results_manager: ResultsManager, data_split_name: str, execution_results_dirpath: str, actor_name: str, actor_uuid: str, submission_epoch_str: str, processed_metrics: list, skip_upload_existing: bool, pre_processed_metrics_dict: dict):
        # Initialize error strings to return
        errors = {}
        new_processed_metric_names = []
        web_display_parse_errors = ''

        # Load results dataframe
        df = self.load_results_df(results_manager)

        # Check to make sure that all metrics exist in the dataframe
        missing_columns = []
        for metric_name, metric in self.submission_metrics.items():
            if metric.store_result and metric_name not in df.columns:
                missing_columns.append(metric_name)

        # Add all missing metric columns, each with default value None
        if len(missing_columns) > 0:
            df = df.assign(**{col: None for col in missing_columns})

        filtered_df = results_manager.filter_primary_key(df, submission_epoch_str, data_split_name, actor_uuid)
        update_entry = {}
        metrics_to_compute = []

        # If the entry already exists, then we need to check for missing/empty metrics
        if filtered_df is not None:
            if len(filtered_df) > 1:
                logging.error('Found {} entries for submission {}, split {}, actor {}'.format(len(filtered_df), submission_epoch_str, data_split_name, actor_name))

            submission_metric_names = self.submission_metrics.keys()

            # Check for metrics to compute
            for metric_name in submission_metric_names:
                if processed_metrics is None or metric_name not in processed_metrics:
                    metrics_to_compute.append(metric_name)
        # Entry is new, so we are creating a new row
        else:
            update_entry = {'submission_timestamp': submission_epoch_str, 'data_split': data_split_name,
                            'actor_UUID': actor_uuid, 'actor_name': actor_name}
            # Add all metrics to compute
            metrics_to_compute.extend(self.submission_metrics.keys())

        if len(metrics_to_compute) > 0:
            # Load ground truth and results
            gt_dict: typing.OrderedDict[str, typing.OrderedDict] = self.get_dataset(data_split_name).load_ground_truth()
            results = self.get_dataset(data_split_name).load_results(execution_results_dirpath)

            web_display_parse_errors += self.get_dataset(data_split_name).get_result_errors(execution_results_dirpath)

            # TODO: Update 'update_entry' to contain columns of interest relevant for for results df
            # TODO: Do we need a metadata_df
            metadata_df = None

            external_share_files = []
            actor_share_files = []

            for metric_name in metrics_to_compute:
                metric = self.submission_metrics[metric_name]
                metric_output = None
                if pre_processed_metrics_dict is not None and metric_name in pre_processed_metrics_dict:
                    metric_output = pre_processed_metrics_dict[metric_name]
                elif isinstance(metric, VLINCSMetric):
                    metric_output = metric.compute(results, gt_dict, metadata_df, actor_name, self.name, data_split_name, execution_results_dirpath)
                else:
                    logging.warning(
                        'Invalid metric type: {}, expected VideoLINCS for leaderboard {}'.format(type(metric), self.name))
                    continue

                if metric_output is not None:
                    new_processed_metric_names.append(metric_name)
                    if metric.store_result:
                        metric_result = metric_output['result']
                        if metric_result is not None:
                            update_entry[metric_name] = metric_result
                        else:
                            logging.warning('{} Metric {} is slated to return a result, but the result was None'.format(self.name, metric_name))

                    files = metric_output['files']

                    if files is not None:
                        if isinstance(files, str):
                            files = [files]

                        if metric.share_with_actor:
                            actor_share_files.extend(files)

                        if metric.share_with_external:
                            external_share_files.extend(files)

            # Update entry or add entry to result dataframe
            if filtered_df is not None:
                df.loc[filtered_df.index[0], update_entry.keys()] = update_entry.values()
            else:
                df.loc[len(df)] = update_entry

        if len(web_display_parse_errors) != 0:
            errors['web_display_parse_errors'] = web_display_parse_errors

        self.update_results_df(results_manager, df)

        return errors, new_processed_metric_names

    def add_dataset(self, test_harness_config: TestHarnessConfig,
                    split_name: str,
                    can_submit: bool,
                    slurm_queue_name: str,
                    slurm_nice: int,
                    generate_metadata_csv: bool=False,
                    on_html: bool=False):
        if self.dataset_manager.has_dataset(split_name):
            raise RuntimeError('Dataset already exists in DatasetManager: {}'.format(split_name))

        dataset = VideoLINCSDataset(test_harness_config, self.name, split_name, can_submit, slurm_queue_name, slurm_nice)
        if self.dataset_manager.add_dataset(dataset):
            # TODO: Decide on metadata CSV
            # if generate_metadata_csv:
            #     self.generate_metadata_csv(overwrite_csv=True)

            self.initialize_html_options(split_name, on_html)
            return True
        return False

    def get_training_dataset_name(self):
        return None

    def update_results_csv(self, result_df: pd.DataFrame, results_manager: ResultsManager, submission_epoch: int, data_split: str, actor_name: str, actor_uuid: str):
        new_data = dict()

        submission_epoch_str = time_utils.convert_epoch_to_iso(submission_epoch)
        df = self.load_results_df(results_manager)
        filtered_df = results_manager.filter_primary_key(df, submission_epoch_str, data_split, actor_uuid)

        result_df_already_exists = result_df is not None and not result_df[
            (result_df['team_name'] == actor_name) & (result_df['submission_timestamp'] == submission_epoch_str) & (
                    result_df['data_split'] == data_split)].empty

        if result_df_already_exists:
            return new_data

        if filtered_df is None or len(filtered_df) != 1:
            logging.warning('Failed to find {}, {}, {}, when generating round results CSV'.format(
                submission_epoch_str, data_split, actor_uuid))
            return new_data

        # TODO: Pull out data from filtered_df to pass to CSV

        return new_data

    def get_valid_metric(self, metric_name):
        return VideoLINCSLeaderboard.VALID_METRIC_NAMES[metric_name]

    def get_valid_summary_metric(self, metric_name):
        raise NotImplementedError()


def init_leaderboard(args):
    test_harness_config = TestHarnessConfig.load_json(args.test_harness_config_filepath)
    required_files = None
    if args.required_files is not None:
        required_files = args.required_files.split(',')

    leaderboard_type = args.leaderboard_type

    leaderboard = None

    if leaderboard_type == 'vlincs':
        leaderboard = VideoLINCSLeaderboard(args.name, args.task_name, test_harness_config, add_default_data_split=args.add_default_datasplit)

    # TODO: Update with correct leaderboard type
    # if leaderboard_type == 'trojai':
    #     leaderboard = TrojAILeaderboard(args.name, args.task_name, test_harness_config, add_default_data_split=args.add_default_datasplit, required_files=required_files)
    # elif leaderboard_type == 'mitigation':
    #     leaderboard = MitigationLeaderboard(args.name, args.task_name, test_harness_config, add_default_data_split=args.add_default_datasplit, required_files=required_files)
    # elif leaderboard_type == 'llm_mitigation':
    #     leaderboard = LLMMitigationLeaderboard(args.name, args.task_name, test_harness_config, add_default_data_split=args.add_default_datasplit, required_files=required_files)

    if leaderboard is not None:
        leaderboard.save_json(test_harness_config)
        print('Created: {}'.format(leaderboard))

def add_dataset_to_leaderboard(args):
    test_harness_config = TestHarnessConfig.load_json(args.test_harness_config_filepath)

    if args.slurm_queue_name is None:
        slurm_queue_name = Leaderboard.GENERAL_SLURM_QUEUE_NAME
    else:
        slurm_queue_name = args.slurm_queue_name

    leaderboard = Leaderboard.load_json(test_harness_config, args.name)

    if args.split_name == 'sts':
        auto_delete_submission = True
    else:
        auto_delete_submission = False

    if leaderboard.add_dataset(test_harness_config, args.split_name, args.can_submit, slurm_queue_name, args.slurm_nice, args.has_source_data, auto_delete_submission, [], on_html=args.on_html):
        leaderboard.generate_metadata_csv(overwrite_csv=True)
        leaderboard.save_json(test_harness_config)

        print('Added dataset {} to {}'.format(args.split_name, args.name))
    else:
        print('Failed to add dataset')


def generate_summary_metadata(args):
    test_harness_config = TestHarnessConfig.load_json(args.test_harness_config_filepath)
    leaderboard = Leaderboard.load_json(test_harness_config, args.name)
    leaderboard.generate_metadata_csv(overwrite_csv=True)


def add_metric(args):
    test_harness_config = TestHarnessConfig.load_json(args.test_harness_config_filepath)
    leaderboard = Leaderboard.load_json(test_harness_config, args.name)
    metric_name = args.metric_name

    metric_params_filepath = args.metric_params_json_filepath
    metric_param_dict = None

    if metric_params_filepath is not None:
        try:
            with open(metric_params_filepath, mode='r', encoding='utf-8') as f:
                metric_param_dict = json.load(f)
        except json.decoder.JSONDecodeError:
            logging.error("JSON decode error for file: {}, is it a proper json?".format(metric_params_filepath))
            raise
        except:
            raise

    if metric_param_dict is None:
        new_metric = leaderboard.get_valid_metric(metric_name)()
    else:
        new_metric = leaderboard.get_valid_metric(metric_name)(**metric_param_dict)

    leaderboard.add_metric(new_metric)
    leaderboard.save_json(test_harness_config)

    print('Added metric {} to {}'.format(metric_name, leaderboard.name))

def remove_metric(args):
    test_harness_config = TestHarnessConfig.load_json(args.test_harness_config_filepath)
    leaderboard = Leaderboard.load_json(test_harness_config, args.name)
    metric_name = args.metric_name

    leaderboard.remove_metric(metric_name)
    leaderboard.save_json(test_harness_config)

    print('Removed metric {} to {}'.format(metric_name, leaderboard.name))

def add_summary_metric(args):
    test_harness_config = TestHarnessConfig.load_json(args.test_harness_config_filepath)
    leaderboard = Leaderboard.load_json(test_harness_config, args.name)
    metric_name = args.metric_name

    metric_params_filepath = args.metric_params_json_filepath
    metric_param_dict = None

    if metric_params_filepath is not None:
        try:
            with open(metric_params_filepath, mode='r', encoding='utf-8') as f:
                metric_param_dict = json.load(f)
        except json.decoder.JSONDecodeError:
            logging.error("JSON decode error for file: {}, is it a proper json?".format(metric_params_filepath))
            raise
        except:
            raise

    if metric_param_dict is None:
        new_metric = leaderboard.get_valid_summary_metric(metric_name)()
    else:
        new_metric = leaderboard.get_valid_summary_metric(metric_name)(**metric_param_dict)

    leaderboard.summary_metrics.append(new_metric)
    leaderboard.save_json(test_harness_config)

    print('Added summary metric {} to {}'.format(metric_name, leaderboard.name))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Runs leaderboards commands')
    parser.set_defaults(func=lambda args: parser.print_help())

    subparser = parser.add_subparsers(dest='cmd', required=True)

    init_parser = subparser.add_parser('init', help='Initializes a new leaderboard')
    init_parser.add_argument('--leaderboard-type', type=str, choices=Leaderboard.LEADERBOARD_TYPES, help='Declares the type of leaderboard to generate')
    init_parser.add_argument('--test-harness-config-filepath', type=str, help='The filepath to the main test harness config', required=True)
    init_parser.add_argument('--name', type=str, help='The name of the leaderboards', required=True)
    init_parser.add_argument('--task-name', type=str, choices=Leaderboard.ALL_TASK_NAMES, help='The name of the task for this leaderboards', required=True)
    init_parser.add_argument('--required-files', type=str, default=None, help='The set of required files, defaults to the defaults set if not used. Tis is a csv list like " --required-files=model.pt,test.sh,img.png"')
    init_parser.add_argument('--add-default-datasplit', help='Will attempt to add the default data splits, if they fail task checks then will not be added. Need to call add-dataset when they are ready.', action='store_true')
    init_parser.set_defaults(func=init_leaderboard)

    add_dataset_parser = subparser.add_parser('add-dataset', help='Adds a dataset into a leaderboard')
    add_dataset_parser.add_argument('--test-harness-config-filepath', type=str, help='The filepath to the main test harness config', required=True)
    add_dataset_parser.add_argument('--name', type=str, help='The name of the leaderboards', required=True)
    add_dataset_parser.add_argument('--split-name', type=str, help='The dataset split name', required=True)
    add_dataset_parser.add_argument('--has-source-data', action='store_true', help='Indicates that the dataset has source data that is saved on disk, format: "leaderboard_name-source_data"')
    add_dataset_parser.add_argument('--can-submit', action='store_true', help='Whether actors can submit to the dataset')
    add_dataset_parser.add_argument('--slurm-queue-name', type=str, help='The name of the slurm queue')
    add_dataset_parser.add_argument('--slurm-nice', type=int, help='The nice value when launching jobs for this dataset (0 is highest priority)', default=0)
    add_dataset_parser.add_argument('--on-html', action='store_true', help='Whether the dataset will be shown on the HTML page')
    add_dataset_parser.set_defaults(func=add_dataset_to_leaderboard)

    summary_results_parser = subparser.add_parser('generate_summary_metadata', help='Generates the METADATA CSV file for the leaderboard')
    summary_results_parser.add_argument('--test-harness-config-filepath', type=str, help='The filepath to the main test harness config',required=True)
    summary_results_parser.add_argument('--name', type=str, help='The name of the leaderboards', required=True)
    summary_results_parser.set_defaults(func=generate_summary_metadata)

    add_metric_parser = subparser.add_parser('add-metric', help='Adds metric to leaderboard')
    add_metric_parser.add_argument('--test-harness-config-filepath', type=str, help='The filepath to the main test harness config', required=True)
    add_metric_parser.add_argument('--name', type=str, help='The name of the leaderboards', required=True)
    add_metric_parser.add_argument('--metric-name', type=str, choices=Leaderboard.ALL_METRIC_NAMES, help='The name of the metric to add', required=True)
    add_metric_parser.add_argument('--metric-params-json-filepath', type=str, help='The filepath to the json file that describes custom metric parameters', default=None)
    add_metric_parser.set_defaults(func=add_metric)

    remove_metric_parser = subparser.add_parser('remove-metric', help='Removes metric from leaderboard')
    remove_metric_parser.add_argument('--test-harness-config-filepath', type=str, help='The filepath to the main test harness config', required=True)
    remove_metric_parser.add_argument('--name', type=str, help='The name of the leaderboards', required=True)
    remove_metric_parser.add_argument('--metric-name', type=str, help='The name of the metric to remove', required=True)
    remove_metric_parser.set_defaults(func=remove_metric)

    add_summary_metric_parser = subparser.add_parser('add-summary-metric', help='Adds a summary metric to leaderboard')
    add_summary_metric_parser.add_argument('--test-harness-config-filepath', type=str, help='The filepath to the main test harness config', required=True)
    add_summary_metric_parser.add_argument('--name', type=str, help='The name of the leaderboards', required=True)
    add_summary_metric_parser.add_argument('--metric-name', type=str, choices=Leaderboard.ALL_SUMMARY_METRIC_NAMES, help='The name of the metric to add', required=True)
    add_summary_metric_parser.add_argument('--metric-params-json-filepath', type=str, help='The filepath to the json file that describes custom metric parameters', default=None)
    add_summary_metric_parser.set_defaults(func=add_summary_metric)

    args = parser.parse_args()
    args.func(args)

