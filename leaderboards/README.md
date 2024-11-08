# Design philosophy

VideoLINCs consists of multiple leaderboards. Each leaderboard targets a specific task, contains a set of datasets, and a set of metrics per dataset that is computed on performer predictions compared with dataset ground truth.

The datasets describe the required files and any files that can be omitted when doing evaluation.

The leaderboard's task is unique per leaderboard, and each dataset should be in-line with that task. Overall the task is responsible for and specializes:
1. Verifying the dataset (contains the required files)
2. Check submissions for necessary requirements
3. Copy in necessary data into VM 
4. Execute the submission
5. Copy out results
6. Clean up the VM for next execution

All components are configured based on a couple of json files:
1. test_harness_config.json -- Configuration for all of the test harness
2. leaderboard configs -- Describes a leaderboard and its datasets
3. actors -- Describes all actors in VLINCs
4. submissions -- Describes the submissions into a leaderboard


## Setup VLINCs Back-end

1. Run test_harness_config.py to create VLINCs configuration json file
2. Run leaderboard.py "init" to create a leaderboard (specify --add-default-datasplit, to add "test, sts, holdout, and train" datasets, should only be done after they are added into the datasets folder, use the name "LEADERBOARD_NAME-SPLIT_NAME-dataset")
3. Run leaderboard.py "add-dataset" to add datasets to a leaderboard (for any datasets )
4. Run actor.py "add-actor" to add an actor

Customize test_harness_config.json, and json files in leaderboard-configs directory.  

## Setup VLINCs Front-end

1. Clone https://github.com/usnistgov/vlincs html   (place into html folder or as configured in test_harness_config.json)
2. git checkout nist-pages (web-hook page)
3. Ensure execution user of check_and_launch_actors.py has permission to push to vlincs repo.


# Setup Drive API Access

Goto the Developer console
https://console.developers.google.com/

- Create a new project "vlincs"

- Enable APIs - Enable Google Drive API

- Configure OAuth2 Screen

  Click on 'Credentials' in the LHS page menu.
  + Click on 'Configure Consent Screen'
  + Select 'Internal' for 'User Type' and click on 'Create'
  + Set the 'Application name' field to 'VLINCs'
  + Add the following scope: '../auth/drive'
  + Click on 'Save'

- Create Credentials (OAuth2)
  + Click on 'Credentials'
  + Click on 'Create Credentials'
  + Choose 'OAuth Client ID'
  + Choose 'Other' as 'Application type'
  + Set 'Name' to 'VLINCs'
  + Click on 'Save' or 'Create'

	Creates 'OAuth client' and gives 'Client ID' and 'Client Secret',
    which look like a public-private key pair.

  + Click OK

- Download credential, rename to credentials.json

-------------------------------------------------------------------------------

# libvert

Install libvert on the system.

`sudo apt install libvirt-dev`


# Setup Python for Google Drive Access

Based on https://developers.google.com/drive/api/v3/quickstart/python


`conda create --name drive python=3.7`

`conda activate drive`

`pip install --upgrade -r requirements.txt`


python3 -m venv test-env
source ~/test-env/bin/activate
pip install --upgrade wheel google-api-python-client google-auth-httplib2 google-auth-oauthlib jsonpickle pid numpy pytablewriter dominate GitPython httplib2==0.15

-------------------------------------------------------------------------------

# Create and Downlaod OAuth Token

- Run the following command; make sure paths are *right*.

```
	pushd vlincs/src/te-scripts/actor_executor
	python3 create_auth_token.py \
	--credentials-filepath ~/Projects/vlincs/OAuth-creds/credentials.json \
	--token-pickle-filepath ~/Projects/vlincs/OAuth-creds/token.pickle
```

# Add Actor

```
python actor_manager.py --add-actor="<team name>,<submitting email>" --config-file=/mnt/isgnas/configRound0.json --log-file=./actor-log
```

# Start Test Loop

- Run the following command; make sure paths are *right*.

```
	pushd vlincs/src/te-scripts/actor_executor
	python3 infinite_submitter.py \
	--token-pickle-filepath ~/Projects/vlincs/OAuth-creds/token.pickle \
	--filepath ~/Projects/vlincs/fake_trojan_detector.sif 
```


-------------------------------------------------------------------------------

# Restrictions for Files Shared with VLINCs Drive


# FAQ

## Which VMs Go Where

STS (slurm queue test) has vms: 61, db

ES (slurm queue production) has vms: 3b, 60, 86, da 

## To share a file with VLINCs Google Drive user

1. Upload the file in question to your Google Drive account
2. Right click on the file and select "Share"
3. Enter "vlincs@nist.gov" and click Done

## To Stop sharing a file

1. Right click on the file in question and select "Share".
2. Click on "Advanced" (bottom right of the dialog box).
3. Remove the people the file is shared with as required.
4. Click "Save changes"

## File Shared with VLINCs Drive user does not show up in VLINCs's Drive 'Shared with me' folder.

If vlincs@nist.gov removes a file shared with it, that file will not show up in the vlincs@nist.gov Drive again. The file will need to be deleted and a new copy uploaded and shared with vlincs@nist.gov. Using the drive_io python tools, if the file exists in the target Google Drive account it will modify that existing file, not replace it, preventing the file from showing up for VLINCs Drive if it was removed on the VLINCs side. Delete the file from the source drive, re-upload a new copy, and share the new copy with vlincs@nist.gov to re-share the file.