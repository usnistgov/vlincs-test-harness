import logging

from submission_file import SubmissionFile
import typing

class SubmissionIO(object):
    VALID_NAMES = ['g_drive', 'filesystem']
    def __init__(self):
        pass

    def get_name(self):
        raise NotImplementedError()

    def query_worker(self, name: str=None, email: str=None, parent_id: str=None, is_folder: bool=False) -> typing.List[SubmissionFile]:
        raise NotImplementedError()

    def download(self, s_file: SubmissionFile, output_dirpath: str) -> None:
        raise NotImplementedError()

    def create_folder(self, folder_name, parent_id='root') -> str:
        raise NotImplementedError()

    def upload(self, file_path: str, folder_id=None, skip_existing=False) -> str:
        raise NotImplementedError()

    def share(self, file_id: str, share_email: str) -> None:
        raise NotImplementedError()

    def remove_all_sharing_permissions(self, file_id: str) -> None:
        raise NotImplementedError()

    def query_folder(self, folder_name: str, parent_id='root') -> typing.List[SubmissionFile]:
        return self.query_worker(name=folder_name, parent_id=parent_id, is_folder=True)

    def query_by_filename(self, file_name: str) -> typing.List[SubmissionFile]:
        return self.query_worker(name=file_name)

    def query_by_email_and_filename(self, email: str, file_name: str, folder_id=None) -> typing.List[SubmissionFile]:
        return self.query_worker(name=file_name, email=email, parent_id=folder_id)

    def query_by_email(self, email: str) -> typing.List[SubmissionFile]:
        return self.query_worker(email=email)

    def create_external_root_folder(self):
        return self.create_folder('admin_leaderboard')

    def create_actor_root_folder(self, actor_name):
        return self.create_folder('results_{}'.format(actor_name))

    def create_leaderboard_summary_folder(self):
        test_harness_summary_folder_id = self.create_external_root_folder()
        return self.create_folder('leaderboard_summary_data', parent_id=test_harness_summary_folder_id)

    def create_actor_summary_folder(self):
        test_harness_summary_folder_id = self.create_external_root_folder()
        return self.create_folder('actor_summary_data', parent_id=test_harness_summary_folder_id)

    def upload_and_share(self, file_path: str, share_email: str | typing.List[str]) -> None:
        file_id = self.upload(file_path)

        # unshare to remove all permissions except for owner, to ensure that if the file is deleted on the receivers end, that they get a new copy of it.
        self.remove_all_sharing_permissions(file_id)

        if isinstance(share_email, list):
            for email in share_email:
                self.share(file_id, email)
        else:
            self.share(file_id, share_email)

    def get_submission_actor_and_external_folder_ids(self, actor_name: str, leaderboard_name: str, data_split_name: str):
        try:
            # Setup the external folder for an actor, leaderboard name, and data split
            actor_plots_folder_id = self.create_actor_summary_folder()
            root_external_folder_id = self.create_folder('{}'.format(actor_name), parent_id=actor_plots_folder_id)
            external_actor_submission_folder_id = self.create_folder('{}_{}'.format(leaderboard_name, data_split_name), parent_id=root_external_folder_id)

            # Setup folder for actor
            root_actor_folder_id = self.create_actor_root_folder(actor_name)
            actor_submission_folder_id = self.create_folder('{}_{}'.format(leaderboard_name, data_split_name), parent_id=root_actor_folder_id)
        except:
            logging.error('Failed to create google drive actor directories')
            actor_submission_folder_id = None
            external_actor_submission_folder_id = None

        return actor_submission_folder_id, external_actor_submission_folder_id

    def submission_download(self, email: str, output_dirpath: str, metadata_filepath: str, requested_leaderboard_name: str, requested_data_split_name) -> SubmissionFile:
        actor_file_list = self.query_by_email(email)

        # filter list based on file prefix
        gdrive_file_list = list()
        for g_file in actor_file_list:
            filename = g_file.name
            filename_split = filename.split('_')

            if len(filename_split) <= 2:
                continue

            leaderboard_name = filename_split[0]
            data_split_name = filename_split[1]

            if leaderboard_name == requested_leaderboard_name and data_split_name == requested_data_split_name:
                gdrive_file_list.append(g_file)


        # ensure submission is unique (one and only one possible submission file from a team email)
        if len(gdrive_file_list) < 1:
            msg = "Actor does not have submission from email {}.".format(email)
            logging.error(msg)
            raise IOError(msg)
        if len(gdrive_file_list) > 1:
            msg = "Actor submitted {} file from email {}.".format(len(gdrive_file_list), email)
            logging.error(msg)
            raise IOError(msg)

        submission = gdrive_file_list[0]
        submission.save_json(metadata_filepath)
        logging.info('Downloading "{}" from Actor "{}" last modified time "{}".'.format(submission.name, submission.email, submission.modified_epoch))
        self.download(submission, output_dirpath)
        return submission