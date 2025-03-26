conda create -n mllm python=3.10 -y
conda activate mllm
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install deepspeed transformers
conda install -c nvidia cuda-compiler
pip install 'accelerate>=0.26.0'
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126