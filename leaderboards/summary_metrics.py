import numpy as np
import pandas as pd
import logging
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import time_utils


class SummaryMetric(object):
    def __init__(self, share_with_collaborators: bool, add_to_html: bool):
        self.shared_with_collaborators = share_with_collaborators
        self.add_to_html = add_to_html

    def compute_and_write_data(self, leaderboard_name: str, data_split_name: str,  metadata_df: pd.DataFrame, results_df: pd.DataFrame, output_dirpath: str):
        pass





