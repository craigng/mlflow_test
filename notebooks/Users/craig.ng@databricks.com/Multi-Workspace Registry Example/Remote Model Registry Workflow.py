# Databricks notebook source
# Install notebook-scoped libraries
dbutils.library.installPyPI("mlflow", "1.8.0")
dbutils.library.installPyPI("tensorflow", "1.14.0")
dbutils.library.installPyPI("keras")
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ##### Demo set-up: log a model
# MAGIC This part is just so we have a logged Run to work with and is not an important part of the multi-workspace workflow.

# COMMAND ----------

dbutils.notebook.run("LogRun", 120)

# COMMAND ----------

# MAGIC %md At this point, you should see a run in the Runs sidebar (click on the "Runs" menu item at the top right corner of the notebook).

# COMMAND ----------

# Programmatically get the Run ID for the demo - normally you'd probably have the run ID
# you want from the UI or from searching based on some criteria
import json
context = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())
experiment_name = context['extraContext']['notebook_path']
import mlflow
runs = mlflow.search_runs(mlflow.get_experiment_by_name(experiment_name).experiment_id)
latest_run_id = runs.sort_values('end_time').iloc[-1]["run_id"]
print(latest_run_id)

# COMMAND ----------

# MAGIC %md ### Set up the Databricks CLI config file on the cluster with credentials for the remote registry.
# MAGIC This set-up is required for registered, managing, or using models.
# MAGIC 
# MAGIC **Note**: Avoid using a shared cluster as other users may be able to use your CLI profile.

# COMMAND ----------

host = dbutils.secrets.get(scope = "registries_craig", key = "host")
token = dbutils.secrets.get(scope = "registries_craig", key = "token")
cli_profile_name = 'registry'
dbutils.fs.put("file:///root/.databrickscfg","[%s]\nhost=%s\ntoken=%s" % (cli_profile_name, host, token), overwrite=True)

# COMMAND ----------

# MAGIC %fs head file:///root/.databrickscfg

# COMMAND ----------

# Now we can specify an MLflow Tracking URI that can be used to 
# initialize an MlflowClient, or the entire environment
TRACKING_URI = "databricks://%s" % cli_profile_name
print(TRACKING_URI)

# COMMAND ----------

# MAGIC %md ### Registering a Model

# COMMAND ----------

# MAGIC %md ##### Copy the model artifacts from the local tracking store to Registry workspace DBFS 

# COMMAND ----------

### USER INPUT
run_id = latest_run_id  # the local run from which to register the model. here we use the latest run created above.
artifact_path = 'model'

# COMMAND ----------

# Here we define some utilities that will help us copy the model artifacts from this workspace to the remote registry workspace.
# The method to use is `copy_artifacts` at the end of the cell.

import os
import posixpath

from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import relative_path_to_artifact_path
from mlflow.utils.rest_utils import http_request_safe
from mlflow.utils.string_utils import strip_prefix
from mlflow.exceptions import MlflowException

def _get_dbfs_endpoint(artifact_uri, artifact_path):
  return "/dbfs/%s/%s" % (strip_prefix(artifact_uri.rstrip('/'), 'dbfs:/'), strip_prefix(artifact_path, '/'))

def _copy_artifact(local_file, artifact_uri, artifact_path=None):
    basename = os.path.basename(local_file)
    if artifact_path:
        http_endpoint = _get_dbfs_endpoint(artifact_uri, posixpath.join(artifact_path, basename))
    else:
        http_endpoint = _get_dbfs_endpoint(artifact_uri, basename)

    host_creds = get_databricks_host_creds('registry')
    print("Copying file to " + http_endpoint + " in registry workspace")
    try:
      if os.stat(local_file).st_size == 0:
          # The API frontend doesn't like it when we post empty files to it using
          # `requests.request`, potentially due to the bug described in
          # https://github.com/requests/requests/issues/4215
          http_request_safe(host_creds, endpoint=http_endpoint, method='POST', data="", allow_redirects=False)
      else:
          with open(local_file, 'rb') as f:
              http_request_safe(host_creds, endpoint=http_endpoint, method='POST', data=f, allow_redirects=False)
    except MlflowException as e:  
      # Note: instead of catching the error here, we could check for the existence of file before trying the copy.
      if "File already exists" in e.message:
        print("File already exists - continuing to the next file.")
        import time
      else:
        throw(e)
  
  
# params:
#   artifact_uri: the base path for the run.
#   artifact_path: the relative path under `artifact_uri` to the model.
def copy_artifacts(artifact_uri, artifact_path):
    local_dir = "/dbfs/%s/%s" % (strip_prefix(artifact_uri.rstrip('/'), 'dbfs:/'), strip_prefix(artifact_path, '/'))
    artifact_path = artifact_path or ''
    for (dirpath, _, filenames) in os.walk(local_dir):
        artifact_subdir = artifact_path
        if dirpath != local_dir:
            rel_path = os.path.relpath(dirpath, local_dir)
            rel_path = relative_path_to_artifact_path(rel_path)
            artifact_subdir = posixpath.join(artifact_path, rel_path)
        for name in filenames:
            file_path = os.path.join(dirpath, name)
            _copy_artifact(file_path, artifact_uri, artifact_subdir)

# COMMAND ----------

from mlflow.tracking import artifact_utils
artifact_uri = artifact_utils.get_artifact_uri(run_id)

copy_artifacts(artifact_uri, artifact_path)

# COMMAND ----------

# MAGIC %md ##### Create an MlflowClient with Tracking URI set to the registry workspace

# COMMAND ----------

from mlflow.tracking import MlflowClient
remote_client = MlflowClient(tracking_uri=TRACKING_URI)

# COMMAND ----------

# MAGIC %md ##### Call register_model() using the remote client, using the new DBFS location as “source”.
# MAGIC Note that the "source" URI, which is part of the metadata for the newly created model version, may contain the original Run ID if you used the default artifact store location when you logged the model from the Run.

# COMMAND ----------

### USER INPUT
model_name = "my_remote_model"

# COMMAND ----------

import posixpath
source = posixpath.join(artifact_uri, artifact_path)  # we preserved the model artifact dbfs path from the source workspace
try: 
  remote_client.create_registered_model(model_name)
except Exception as e:
  if e.error_code == 'RESOURCE_ALREADY_EXISTS':
    print(e)
  else: 
    throw(e)
mv = remote_client.create_model_version(model_name, source, run_id)  # `source` must point to the DBFS location in the new workspace
print(mv)

# COMMAND ----------

# MAGIC %md At this point, if you log into the registry workspace you should see the new model version.

# COMMAND ----------

# MAGIC %md ##### Optionally, write `<SourceWorkspaceId>` and `<RunID>` as the model version description for lineage tracking.
# MAGIC The `Source Run` field will not show up on the model version page in the registry workspace because it is known to the workspace.

# COMMAND ----------

# Note: if you are okay with exposing the experiment ID to those who can read the model version in the registry workspace, 
# you can create the full URL for the Run with: 
#   %s/_mlflow/?o=%s#/experiments/%s/runs/%s." % (workspace_url, workspace_id, experiment_id, run_id)
# e.g. https://westus.azuredatabricks.net/_mlflow/?o=12345678901234#/experiments/11223344556677/runs/abcd123564dedr136176daiw1234

import json
context = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())
workspace_url = context['extraContext']['api_url']
workspace_id = context['tags']['orgId']
description = "Remote source workspace: %s/?o=%s, run: %s." % (workspace_url, workspace_id, run_id)
print(description)

# COMMAND ----------

remote_client.update_model_version(model_name, mv.version, description=description)

# COMMAND ----------

# MAGIC %md ### Managing a Model

# COMMAND ----------

# For example, we can transition a remote model version to "Production". Note that this action 
# requires the user (token) in the registry workspace to have "Manage" access to the model and
# that the model version is in 'READY' status.

# A hack to wait for the model version to become ready (status) - it'd be better 
# to check in a loop that the model version status = READY
import time 
time.sleep(15)  

remote_client.update_model_version(model_name, mv.version, stage='Production')

# COMMAND ----------

# MAGIC %md ### Using a Model
# MAGIC 
# MAGIC The easiest way to load the model is using the `<module>.load_model` method, which is shown below. Alternatively you can download the file directly from DBFS after getting the URI via `remote_client.get_model_version_download_uri`.

# COMMAND ----------

import mlflow
# Note: Even for the registration workflow above, this mechanism for setting the tracking URI 
# can be used instead of instantiating an MlflowClient with tracking_uri. To reset the tracking 
# URI to point to the current workspace, you can call `mlflow.set_tracking_uri('databricks')`.
mlflow.set_tracking_uri(TRACKING_URI)

from mlflow import keras
loaded_model = keras.load_model('models:/%s/Production' % model_name)

# COMMAND ----------

import mlflow

# COMMAND ----------

from mlflow.utils import mlflow_tags

# COMMAND ----------

client = mlflow.tracking.MlflowClient()

# COMMAND ----------

full_run = client.get_run("aed2b4090e8f4d2a859d56c6e41b47b3")

# COMMAND ----------

tags = full_run.data.tags

# COMMAND ----------

tags

# COMMAND ----------

tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)

# COMMAND ----------

