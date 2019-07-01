# ATPC-Seminar: Data-parallel Training of Neural Networks
Install required packages
```
pip3 install -r requirements.txt
```

Create datasets (download from torchvision to '../datasets' if necessary and convert to HDF5 file)
```
python3 createDatasets.py
```

Run training with
```
python3 main.py
```
Will store experimental results (running times, losses, error rates,...) in two files using the program start time as name:
```
outputs/results__YYYY-mm-dd--HH-MM-SS__epochs
outputs/results__YYYY-mm-dd--HH-MM-SS__summary
```
Configuration options can be displayed with
```
python3 main.py -h
```