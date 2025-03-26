conda create -n mllm python=3.10 -y
conda activate mllm
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install deepspeed transformers
conda install -c nvidia cuda-compiler