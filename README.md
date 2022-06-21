### Re-implementation for ReMixMatch

- Uses WideResNet-28-2 as backbone network
- Uses STL-10 dataset

### Installing environment

- (Recommand) With Docker
Use official Pytorch docker image: `pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel`
Install EfficientNet, tqdm: `pip install tqdm`

- Without Docker
Install Pytorch and tqdm: `pip install torch==1.8.0 torchvision==0.9.0 tqdm`

### Get trained model weight

- Install gdown: `pip install gdown`
- Download weight:
    - #### WideResNet-28-2

### Train on STL-10 dataset
- #### Apply mixmatch's batch computing algorithm to each mini-batch  
    `python train_remixmatch.py`
    - You must specify paths that saves the model checkpoint (`--results_dir`) and training log (`--tensorboard_path`).
