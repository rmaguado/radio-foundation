conda create -n mllm python=3.10 -y
conda activate mllm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu126
pip install transformers 'accelerate>=0.26.0'
conda install -c nvidia cuda-compiler
pip install deepspeed
