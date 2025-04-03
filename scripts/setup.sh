conda create -n radio python=3.10 -y
conda activate radio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu124
pip install transformers 'accelerate>=0.26.0'
pip install flash-attn --no-build-isolation
pip install peft
conda install nvidia/label/cuda-12.4.1::cuda-compiler
pip install deepspeed