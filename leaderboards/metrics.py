# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import json

import numpy as np
import sklearn.metrics
import pandas as pd
import logging

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from typing import Dict, List, Optional, Union

import os

from leaderboards import fs_utils




class Metric(object):
    def __init__(self, write_html: bool, share_with_actor: bool, store_result: bool,
                 share_with_external: bool):
        self.write_html = write_html
        self.share_with_actor = share_with_actor
        self.store_result = store_result
        self.share_with_external = share_with_external
        self.html_priority = 0
        self.html_decimal_places = 5

    def get_name(self):
        raise NotImplementedError()

    # asc or desc for ascending or descending, respectively
    def get_sort_order(self):
        return 'desc'

    def compare(self, computed, baseline):
        raise NotImplementedError()
