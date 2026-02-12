import os

from huggingface_hub import snapshot_download

local_pth = os.environ.get("HF_DATASETS_CACHE")

snapshot_download(
    repo_id="amphion/Emilia-Dataset",
    repo_type="dataset",
    allow_patterns="Emilia/ZH/*",  # 只允許下載中文目錄
    local_dir= local_pth,
    local_dir_use_symlinks=False,
    token=True
)