conda create -n pcc python=3.7
conda activate pcc
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install hdf5storage
pip install Ninja
pip install tensorboard
pip install h5py
pip install tqdm
pip install matplotlib
pip install plyfile
