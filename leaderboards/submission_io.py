import logging
import time

from submission_file import SubmissionFile
import typing
import mimetypes
import shutil
import os
import time_utils
import random
import io
from test_harness_config import TestHarnessConfig
import socket
socket.setdefaulttimeout(120)

def init_submission_io(submission_io_str: str, test_harness_config: TestHarnessConfig) -> 'SubmissionIO':
    submission_io = None
    if submission_io_str == 'g_drive':
        submission_io = DriveIO(test_harness_config.token_pickle_filepath)
    elif submission_io_str == 'filesystem':
        submission_io = FileIO(test_harness_config.test_harness_dirpath)
    else:
        logging.error('Invalid submission system specified: {}'.format(submission_io_str))
        raise RuntimeError('Invalid submission system specified: {}'.format(submission_io_str))

    return submission_io

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

        if not os.path.exists(query_dirpath):
            os.makedirs(query_dirpath, exist_ok=True)
            logging.info('Creating query directory: {}'.format(query_dirpath))

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


logging.getLogger('googleapiclient.discovery').setLevel(logging.WARNING)
logging.getLogger('google_auth_oauthlib.flow').setLevel(logging.WARNING)
logging.getLogger('google.auth.transport.requests').setLevel(logging.WARNING)
logging.getLogger('googleapiclient.http').setLevel(logging.WARNING)
logging.getLogger('googleapiclient.errors').setLevel(logging.WARNING)



class DriveIO(SubmissionIO):
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/drive']

    def __init__(self, token_pickle_filepath):
        super().__init__()
        self.token_pickle_filepath = token_pickle_filepath
        self.page_size = 100
        self.max_retry_count = 4
        # self.request_count = 0

        self.__get_service(self.token_pickle_filepath)

        self.folder_cache = {}
        self.folder_times = {}
        # Cache is stale after 120 seconds
        self.stale_time_limit = 120

    def get_name(self):
        return 'g_drive'

    def __get_service(self, token_json_filepath):
        from googleapiclient.discovery import build
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        logging.debug('Starting connection to Google Drive.')
        creds = None
        try:
            # The file token.pickle stores the user's access and refresh tokens, and is
            # created automatically when the authorization flow completes for the first
            # time.
            if os.path.exists(token_json_filepath):
                creds = Credentials.from_authorized_user_file(token_json_filepath, DriveIO.SCOPES)

            logging.debug('Token credentials loaded')
            # If there are no (valid) credentials available, let the user log in.
            if not creds:
                logging.error('Credentials could not be loaded. Rebuild token using create_auth_token.py')
                raise RuntimeError('Credentials could not be loaded. Rebuild token using create_auth_token.py')

            # check if the credentials are not valid
            if not creds.valid:
                if creds.expired and creds.refresh_token:
                    logging.debug('Credentials exists, but are no longer valid, attempting to refresh.')
                    creds.refresh(Request())
                    logging.debug('Credentials refreshed successfully.')
                    # Save the credentials for the next run
                    with open(token_json_filepath, 'w') as token:
                        token.write(creds.to_json())
                    logging.debug('Credentials refreshed and saved to "{}".'.format(token_json_filepath))
                else:
                    logging.error('Could not refresh credentials. Rebuild token using create_auth_token.py.')
                    raise RuntimeError('Could not refresh credentials. Rebuild token using create_auth_token.py.')

            logging.debug('Building Drive service from credentials.')
            self.service = build('drive', 'v3', credentials=creds, cache_discovery=False)
            # Turn off cache discover to prevent logging warnings
            # https://github.com/googleapis/google-api-python-client/issues/299

            logging.debug('Querying Drive to determine account owner details.')
            response = self.service.about().get(fields="user").execute(num_retries=self.max_retry_count)
            # self.request_count += 1
            self.user_details = response.get('user')
            self.email_address = self.user_details.get('emailAddress')
            logging.info('Connected to Drive for user: "{}" with email "{}".'.format(self.user_details.get('displayName'), self.email_address))
        except Exception as e:
            logging.error('Failed to connect to Drive.')
            logging.error('Exception: {}'.format(e))
            raise

    def query_worker(self, name: str=None, email: str=None, parent_id: str=None, is_folder: bool=False) -> typing.List[SubmissionFile]:
        from googleapiclient.errors import HttpError

        query = None

        query_list = []
        if name is not None:
            query_list.append("name = '{}' and trashed = false".format(name))

        if email is not None:
            query_list.append("'{}' in owners".format(email))

        if parent_id is not None:
            query_list.append("'{}' in parents".format(parent_id))

        if is_folder:
            query_list.append("mimeType = '{}'".format("application/vnd.google-apps.folder"))

        if len(query_list) == 0:
            logging.error('The query list is empty, call to query worker incorrect')
            raise RuntimeError()
        elif len(query_list) == 1:
            query = query_list[0]
        else:
            query = " and ".join(query_list)


        # https://developers.google.com/drive/api/v3/search-files
        # https://developers.google.com/drive/api/v3/reference/query-ref
        try:
            logging.debug('Querying Drive API with "{}".'.format(query))
            retry_count = 0
            while True:
                try:
                    # Call the Drive v3 API, blocking through pageSize records for each call
                    page_token = None
                    items = list()
                    while True:
                        # name, id, modifiedTime, sharingUser
                        response = self.service.files().list(q=query,
                                                             pageSize=self.page_size,
                                                             fields="nextPageToken, files(name, id, modifiedTime, owners)",
                                                             pageToken=page_token,
                                                             spaces='drive').execute(num_retries=self.max_retry_count)
                        # self.request_count += 1
                        items.extend(response.get('files'))
                        page_token = response.get('nextPageToken', None)
                        if page_token is None:
                            break
                    break  # successfully download file list, break exponential backoff scheme loop
                except HttpError as e:
                    retry_count = retry_count + 1
                    if e.resp.status in [104, 404, 408, 410] and retry_count <= self.max_retry_count:
                        # Start the upload from the beginning.
                        logging.info('Drive Query Error, restarting query from beginning (attempt {}/{}) with exponential backoff.'.format(retry_count, self.max_retry_count))
                        sleep_time = random.random() * 2 ** retry_count
                        time.sleep(sleep_time)
                    else:
                        raise

            logging.debug('Downloaded list of {} files from Drive account.'.format(len(items)))
            file_list = list()
            for item in items:
                owner = item['owners'][0]  # user first owner by default
                g_file = SubmissionFile(owner['emailAddress'], item['name'], item['id'], item['modifiedTime'])
                file_list.append(g_file)
        except:
            logging.error('Failed to connect to and list files from Drive.')
            raise

        return file_list

    def download(self, s_file: SubmissionFile, output_dirpath: str) -> None:
        from googleapiclient.http import MediaIoBaseDownload
        from googleapiclient.errors import HttpError

        retry_count = 0
        logging.info('Downloading file: "{}" from Drive'.format(s_file))
        while True:
            try:
                request = self.service.files().get_media(fileId=s_file.id)
                # self.request_count += 1
                file_data = io.FileIO(os.path.join(output_dirpath, s_file.name), 'wb')
                downloader = MediaIoBaseDownload(file_data, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk(num_retries=self.max_retry_count)
                    logging.debug("  downloaded {:d}%".format(int(status.progress() * 100)))
                return  # download completed successfully
            except HttpError as e:
                retry_count = retry_count + 1
                if e.resp.status in [104, 404, 408, 410] and retry_count <= self.max_retry_count:
                    # Start the upload from the beginning.
                    logging.info('Download Error, restarting download from beginning (attempt {}/{}) with exponential backoff.'.format(retry_count, self.max_retry_count))
                    sleep_time = random.random() * 2 ** retry_count
                    time.sleep(sleep_time)
                else:
                    raise
            except:
                logging.error('Failed to download file "{}" from Drive.'.format(s_file.name))
                raise

    def create_folder(self, folder_name, parent_id='root') -> str:
        from googleapiclient.errors import HttpError
        folder_key = '{}_{}'.format(folder_name, parent_id)
        if folder_key in self.folder_cache.keys():
            # Check staleness
            if folder_key in self.folder_times.keys():
                folder_time = self.folder_times[folder_key]

                cur_epoch = time_utils.get_current_epoch()
                if folder_time + self.stale_time_limit > cur_epoch:
                    return self.folder_cache[folder_key]
                else:
                    del self.folder_cache[folder_key]
                    del self.folder_times[folder_key]

        file_metadata = {
            'name': folder_name,
            'parents': [parent_id],
            'mimeType': 'application/vnd.google-apps.folder'
        }

        retry_count = 0
        logging.debug('Creating google drive folder {}'.format(folder_name))

        while True:
            try:
                existing_folders = self.query_folder(folder_name, parent_id=parent_id)

                if len(existing_folders) > 0:
                    return existing_folders[0].id

                file = self.service.files().create(body=file_metadata, fields='id').execute()
                # self.request_count += 1
                self.folder_cache[folder_key] = file.get('id')
                self.folder_times[folder_key] = time_utils.get_current_epoch()
                return file.get('id')

            except HttpError as e:
                retry_count = retry_count + 1
                if e.resp.status in [104, 404, 408, 410] and retry_count <= self.max_retry_count:
                    logging.info('Folder creation error, restarting (attempt {}/{}) with exponential backoff'.format(retry_count, self.max_retry_count))
                    slee_time = random.random() * 2 ** retry_count
                    time.sleep(slee_time)
                else:
                    raise
            except:
                logging.error('Failed to create folder  "{}" from Drive.'.format(folder_name))
                raise

    def upload(self, file_path: str, folder_id=None, skip_existing=False) -> str:
        from googleapiclient.http import MediaFileUpload
        from googleapiclient.errors import HttpError

        _, file_name = os.path.split(file_path)
        logging.info('Uploading file: "{}" to Drive'.format(file_name))
        m_type = mimetypes.guess_type(file_name)[0]

        # ensure file_path is a regular file
        if not os.path.isfile(file_path):
            logging.error('Upload file_path = "{}" is not a regular file, aborting upload.'.format(file_path))
            raise RuntimeError('Upload file_path = "{}" is not a regular file, aborting upload.'.format(file_path))

        for retry_count in range(self.max_retry_count):
            try:
                existing_files_list = self.query_by_email_and_filename(self.email_address, file_name, folder_id=folder_id)

                existing_file_id = None
                if len(existing_files_list) > 0:
                    existing_file_id = existing_files_list[0].id

                if existing_file_id is not None and skip_existing:
                    logging.info('Skipping upload {}, it already exists on gdrive'.format(file_name))
                    return existing_file_id

                if folder_id is None:
                    file_metadata = {'name': file_name}
                else:
                    if existing_file_id is not None:
                        file_metadata = {'name': file_name}
                    else:
                        file_metadata = {'name': file_name, 'parents': [folder_id]}

                media = MediaFileUpload(file_path, mimetype=m_type, resumable=True)

                if existing_file_id is not None:
                    logging.info("Updating existing file '{}' on Drive.".format(file_name))
                    request = self.service.files().update(fileId=existing_file_id, body=file_metadata, media_body=media)
                    # self.request_count += 1
                else:
                    logging.info("Uploading new file '{}' to Drive.".format(file_name))
                    request = self.service.files().create(body=file_metadata, media_body=media, fields='id')
                    # self.request_count += 1

                response = None
                # loop while there are additional chunks
                while response is None:
                    status, response = request.next_chunk(num_retries=self.max_retry_count)
                    if status:
                        logging.debug("  uploaded {:d}%".format(int(status.progress() * 100)))

                file = request.execute()
                return file.get('id')  # upload completed successfully

            except HttpError as e:
                if e.resp.status in [104, 404, 408, 410] and retry_count <= self.max_retry_count:
                    # Start the upload from the beginning.
                    logging.info('Upload Error, restarting upload from beginning (attempt {}/{}) with exponential backoff.'.format(retry_count, self.max_retry_count))
                    sleep_time = random.random() * 2 ** retry_count
                    time.sleep(sleep_time)
                else:
                    raise
            except:
                logging.error("Failed to upload file '{}' to Drive.".format(file_name))
                raise

    def share(self, file_id: str, share_email: str) -> None:
        if share_email is not None:
            # update the permissions to share the log file with the team, using short exponential backoff scheme
            user_permissions = {'type': 'user', 'role': 'reader', 'emailAddress': share_email}
            for retry_count in range(self.max_retry_count):
                if retry_count == 0:
                    sleep_time = 0.1
                else:
                    sleep_time = random.random() * 2 ** retry_count
                time.sleep(sleep_time)
                try:
                    self.service.permissions().create(fileId=file_id, body=user_permissions, fields='id', sendNotificationEmail=False).execute()
                    # self.request_count += 1
                    logging.info('Successfully shared file {} with {}.'.format(file_id, share_email))
                    return  # permissions were successfully modified if no exception
                except:
                    if retry_count <= 4:
                        logging.info('Failed to modify permissions on try, performing random exponential backoff.')
                    else:
                        logging.error("Failed to share uploaded file '{}' with '{}'.".format(file_id, share_email))
                        raise

    def remove_all_sharing_permissions(self, file_id: str) -> None:
        permissions = self.service.permissions().list(fileId=file_id).execute()
        # self.request_count += 1
        permissions = permissions['permissions']

        for permission in permissions:
            if permission['role'] != 'owner':
                for retry_count in range(self.max_retry_count):
                    if retry_count == 0:
                        sleep_time = 0.1
                    else:
                        sleep_time = random.random() * 2 ** retry_count
                    time.sleep(sleep_time)

                    try:
                        self.service.permissions().delete(fileId=file_id, permissionId=permission['id']).execute()
                        # self.request_count += 1
                        logging.info("Successfully removed share permission '{}' from file {}.".format(permission, file_id))
                        break  # break retry loop
                    except:
                        if retry_count <= 4:
                            logging.info('Failed to modify permissions on try, performing random exponential backoff.')
                        else:
                            logging.error("Failed to remove share permission '{}' from file '{}'.".format(permission, file_id))
                            raise
