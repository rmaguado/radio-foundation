conda create -n radio python=3.10 -y
conda activate radio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu126
pip install transformers 'accelerate>=0.26.0'
conda install -c nvidia cuda-compiler
pip install deepspeed
pip install flash-attn --no-build-isolation