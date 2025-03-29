conda create -n dino2 python=3.10 -y
conda activate dino2
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126
conda install -c conda-forge ipykernel ipywidgets
pip3 install -r requirements.txt
