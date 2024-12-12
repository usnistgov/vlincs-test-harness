# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
import logging
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import typing

import numpy as np
import pandas as pd
from vlincs_track_eval_dataset import VLINCSTrackEvalDataset
import trackeval

class Metric(object):
    def __init__(self, write_html: bool, share_with_actor: bool, store_result: bool,
                 share_with_external: bool):
        self.write_html = write_html
        self.share_with_actor = share_with_actor
        self.store_result = store_result
        self.share_with_external = share_with_external
        self.html_priority = 0
        self.html_decimal_places = 5

    def get_result_keys(self):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    # asc or desc for ascending or descending, respectively
    def get_sort_order(self):
        return 'desc'

    def compare(self, computed, baseline, result_key_name=None):
        raise NotImplementedError()

class VLINCSMetric(Metric):

    def __init__(self, write_html: bool, share_with_actor: bool, store_result: bool,
                 share_with_external: bool):
        super().__init__(write_html, share_with_actor, store_result, share_with_external)

    # Returns a dictionary with the following:
    # 'result': None, or dictionary based on get_result_keys
    # 'files': None or list of files saved
    def compute(self, results_dict: typing.Dict[str, pd.DataFrame], ground_truth_dict: typing.Dict[str, typing.OrderedDict], metadata_df: pd.DataFrame,
                actor_name: str, leaderboard_name: str, data_split_name: str,
                output_dirpath: str):
        raise NotImplementedError()


class TrackEvalMetric(VLINCSMetric):
    def __init__(self):
        super().__init__(True, False, True, False)

        self.metric_result_dict = {
            'HOTA': ['HOTA(0)', 'LocA(0)', 'AssA', 'DetA', 'AssRe', 'AssPr', 'DetRe', 'DetPr', 'LocA'],
            'CLEAR': ['MOTA', 'CLR_FN', 'CLR_FP', 'MT', 'ML', 'CLR_Re', 'CLR_Pr', 'FP_per_frame', 'Frag'],
            'Identity': ['IDF1']}

        self.track_eval_metric_lookup = {'HOTA': trackeval.metrics.HOTA,
                                         'CLEAR': trackeval.metrics.CLEAR,
                                         'Identity': trackeval.metrics.Identity}

        self.metric_naming_order = {'MOTA': 'MOTA',
                                    'IDF1': 'IDF1',
                                    'HOTA(0)': 'HOTA',
                                    'MT': 'MT',
                                    'ML': 'ML',
                                    'CLR_FP': 'FP',
                                    'CLR_FN': 'FN',
                                    'CLR_Re': 'Rcll',
                                    'CLR_Pr': 'Prcn',
                                    'AssA': 'AssA',
                                    'DetA': 'DetA',
                                    'AssRe': 'AssRe',
                                    'AssPr': 'AssPr',
                                    'DetRe': 'DetRe',
                                    'DetPr': 'DetPr',
                                    'LocA': 'LocA',
                                    'FP_per_frame': 'FAF',
                                    'Frag': 'Frag'}


        self.metric_threshold = 0.5

    def get_result_keys(self):
        return list(self.metric_naming_order.values())

    def get_name(self):
        return 'TrackEvalMetric'

    # TODO: Update per metric name
    def compare(self, computed, baseline, result_key_name=None):
        return computed[result_key_name] > baseline[result_key_name]

    def compute(self, results_dict: typing.Dict[str, pd.DataFrame], ground_truth_dict: typing.Dict[str, typing.OrderedDict], metadata_df: pd.DataFrame,
                actor_name: str, leaderboard_name: str, data_split_name: str,
                output_dirpath: str):
        compute_ret = {'result': {}, 'files': None}
        result_ret = {}

        track_eval_dataset = VLINCSTrackEvalDataset(actor_name, output_dirpath, results_dict, ground_truth_dict)
        metrics_config = {'THRESHOLD': self.metric_threshold}
        default_eval_config = trackeval.Evaluator.get_default_eval_config()
        default_eval_config['DISPLAY_LESS_PROGRESS'] = False

        evaluator = trackeval.Evaluator(default_eval_config)
        dataset_list = [track_eval_dataset]

        metrics_list = []
        for metric_name in self.metric_result_dict.keys():
            metrics_list.append(self.track_eval_metric_lookup[metric_name](metrics_config))

        if len(metrics_list) == 0:
            raise Exception('No metrics selected for evaluation for {}'.format(self.get_name()))

        output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

        output_result = output_msg['VLINCSTrackEvalDataset'][actor_name]

        if output_result != 'Success':
            logging.warning('TrackEval failed due to: {}'.format(output_result))

        for metric_name, metric_sub_names in self.metric_result_dict.items():
            for metric_sub_name in metric_sub_names:
                metric_result = output_res['VLINCSTrackEvalDataset'][actor_name]['COMBINED_SEQ']['person'][metric_name][metric_sub_name]
                if isinstance(metric_result, np.ndarray):
                    metric_result = float(np.average(metric_result).item())
                else:
                    metric_result = float(metric_result)

                result_ret[metric_sub_name] = metric_result

        final_result_ret = {}

        # Apply ordering
        for metric_sub_name in self.metric_naming_order.keys():
            final_result_ret[self.metric_naming_order[metric_sub_name]] = result_ret[metric_sub_name]

        # TODO: Handle plots generated...
        return {'result': final_result_ret, 'files': None}


