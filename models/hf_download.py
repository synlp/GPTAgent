import os

# feel free to add more models to the repo_list
repo_list = [
    'nlpconnect/vit-gpt2-image-captioning',
    'microsoft/trocr-base-printed'
]

for repo in repo_list:
    if not os.path.exists(repo):
        os.makedirs(repo)
    snapshot_download(repo_id=repo, local_dir=repo, local_dir_use_symlinks=False, ignore_patterns=["*.safetensors"])
