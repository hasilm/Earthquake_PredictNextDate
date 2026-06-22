from huggingface_hub import HfApi
import os

Folder_name="Fearthquake_predictNextDate"
HF_username="hasilm1"
App_name="earthquake_predictNextDate"

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path=str(Folder_name)+"/deployment",
    repo_id=str(HF_username)+"/"+str(App_name), # enter the Hugging Face username here
    repo_type="space",
    path_in_repo=""                          # optional: subfolder path inside the repo
)
