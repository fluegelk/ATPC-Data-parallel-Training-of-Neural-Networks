# ATPC-Seminar: Data-parallel Training of Neural Networks

### Setup
Install required packages
```
pip3 install -r requirements.txt
```
Additionally, it might be necessary to install PyTorch manually for the correct CUDA version, see https://pytorch.org/get-started/locally/.

Create datasets (download from torchvision to '../datasets' if necessary and convert to HDF5 file)
```
python3 datasets.py
```

### Training
Run training with
```
python3 main.py
```
Configuration options can be displayed with
```
python3 main.py -h
```
By default, the experimental results (running times, losses, error rates,...) will be saved in two files using the program start time and a python unique UUID as name:
```
outputs/results__YYYY-mm-dd--HH-MM-SS__UUID__epochs
outputs/results__YYYY-mm-dd--HH-MM-SS__UUID__summary
```

Run with p processes using
```
mpiexec -n p python3 main.py
```