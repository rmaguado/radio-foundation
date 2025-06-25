source $HOME/.conda_rc

export TRITON_CACHE_DIR=$HOME/.triton

# customize
# mkdir -p /scratch/VM/radio-foundation/.triton 
# ln -s /scratch/VM/radio-foundation/.triton $HOME/.triton

conda create -n radio python=3.10 -y
conda activate radio
conda install -c conda-forge libaio -y
conda install -c nvidia/label/cuda-12.6.3 cuda -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r ../requirements.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu126
pip install transformers 'accelerate>=0.26.0'
pip install flash-attn --no-build-isolation
pip install peft
pip install nvitop
export LDFLAGS="$LDFLAGS -L $HOME/.conda/envs/deeps/lib"
export CFLAGS="$CFLAGS -L $HOME/.conda/envs/deeps/include"
DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 pip install deepspeed --no-cache-dir 

ds_report
