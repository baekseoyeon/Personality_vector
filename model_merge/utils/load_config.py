import torch


if torch.cuda.is_available():
    cache_dir = "/User/dssalpc/Desktop/model_merge/.cache"
else:
    cache_dir = "/Users/yule/.cache"
