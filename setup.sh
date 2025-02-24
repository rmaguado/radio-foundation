conda create -n radio python=3.10 -y
conda activate radio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge omegaconf torchmetrics fvcore iopath submitit -y
pip3 install --extra-index-url https://pypi.nvidia.com cuml-cu12
pip3 install -r requirements.txt
