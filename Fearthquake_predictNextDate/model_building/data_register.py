from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

HF_username="hasilm1"
App_name="earthquake_predictNextDate"
Folder_name="Fearthquake_predictNextDate"

repo_id = str(HF_username)+"/"+str(App_name)                         # enter the Hugging Face username here
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path=str(Folder_name)+"/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
