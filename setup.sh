conda create -n dino2t24 python=3.10 -y
conda activate dino2t24
conda install pytorch==2.4.1 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install -c xformers xformers -y
conda install -c conda-forge omegaconf torchmetrics fvcore iopath submitit -y
pip3 install --extra-index-url https://pypi.nvidia.com cuml-cu12
pip3 install nibabel pydicom