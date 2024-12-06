from leaderboards import time_utils
from leaderboards.submission import Submission
from submission_io import SubmissionIO
from submission_file import SubmissionFile
import typing
import os
import shutil

class FileIO(SubmissionIO):
    def __init__(self, root_dirpath: str):
        super().__init__()

        self.root_dirpath = root_dirpath
        self.submission_dirpath = os.path.join(self.root_dirpath, 'file_submissions')

        if not os.path.exists(self.submission_dirpath):
            os.makedirs(self.submission_dirpath)

    def get_name(self):
        return 'filesystem'


    def query_worker(self, name: str=None, email: str=None, parent_id: str=None, is_folder: bool=False) -> typing.List[SubmissionFile]:
        # Search for file based on "name"
        sub_dirs = []
        search_name = ''

        if parent_id is not None:
            sub_dirs.append(parent_id)

        if email is not None:
            sub_dirs.append(email)

        if name is not None:
            search_name = name

        if len(sub_dirs) > 0:
            query_dirpath = os.path.join(self.submission_dirpath, *sub_dirs)
        else:
            query_dirpath = self.submission_dirpath

        file_list = os.listdir(query_dirpath)

        submission_file_list = []
        for file in file_list:
            found_filepath = os.path.join(query_dirpath, file)
            modified_time = time_utils.convert_epoch_to_iso(os.path.getmtime(found_filepath))
            submission_file_list.append(SubmissionFile(email, file, query_dirpath, modified_time))


        return submission_file_list


    def download(self, s_file: SubmissionFile, output_dirpath: str) -> None:
        filepath = os.path.join(s_file.id, s_file.name)
        shutil.copy(filepath, output_dirpath)

    def create_folder(self, folder_name, parent_id='root') -> str:
        parent_dirpath = self.submission_dirpath
        if parent_id != 'root':
            parent_dirpath = parent_id

        new_folder_dirpath = os.path.join(parent_dirpath, folder_name)
        os.makedirs(new_folder_dirpath, exist_ok=True)
        return new_folder_dirpath


    def upload(self, file_path: str, folder_id=None, skip_existing=False) -> str:
        return None

    def share(self, file_id: str, share_email: str) -> None:
        pass

    def remove_all_sharing_permissions(self, file_id: str) -> None:
        pass