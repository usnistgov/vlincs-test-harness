from submission_io import SubmissionIO
from fs_io import FileIO
from drive_io import DriveIO
from test_harness_config import TestHarnessConfig
import logging

def init_submission_io(submission_io_str: str, test_harness_config: TestHarnessConfig) -> SubmissionIO:
    submission_io = None
    if submission_io_str == 'g_drive':
        submission_io = DriveIO(test_harness_config.token_pickle_filepath)
    elif submission_io_str == 'filesystem':
        submission_io = FileIO(test_harness_config.test_harness_dirpath)
    else:
        logging.error('Invalid submission system specified: {}'.format(submission_io_str))
        raise RuntimeError('Invalid submission system specified: {}'.format(submission_io_str))

    return submission_io