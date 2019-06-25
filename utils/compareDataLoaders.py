import torch
import torchvision
import numpy as np
import time
import sys

sys.path.append('../implementation/')
import main as m


def iterate(dataloader):
    new_value = np.random.rand(1)[0]
    for i, data in enumerate(dataloader, 0):
        images, labels = data
        for image in images:
            pass
            # image[0][0][0] = new_value


def measure_iteration_time(dataloader, sample_size, print_all=False):
    time_sum = 0
    for i in range(sample_size + 2):
        start = time.time()
        iterate(dataloader)
        end = time.time()

        if i == 0 or i > sample_size:
            continue
        time_sum += end - start
        if print_all:
            print("Time: {:.4f}s".format(end - start))
    print("Avg Time: {:.4f}s".format(time_sum / sample_size))


def measure_dataloader(dlType, batch_size, device, dataset, sample_size, print_all=False, dataset_path="../datasets/"):
    path = dataset_path if dlType is m.DataLoader.Torch else (dataset_path + dataset.name + ".hdf5")
    dataloader, _, _ = m.createDataLoaders(dlType, path, batch_size, device, dataset)
    measure_iteration_time(dataloader, sample_size, print_all)


def main():
    sample_size = 5
    batch_size = 32
    dataset = m.DataSet.MNIST

    print("CPU")
    device_cpu = torch.device('cpu')
    print("Torch:")
    measure_dataloader(m.DataLoader.Torch, batch_size, device_cpu, dataset, sample_size)
    print("SequentialHDF5:")
    measure_dataloader(m.DataLoader.SequentialHDF5, batch_size, device_cpu, dataset, sample_size)
    print("ParallelHDF5:")
    measure_dataloader(m.DataLoader.ParallelHDF5, batch_size, device_cpu, dataset, sample_size)

    if torch.cuda.is_available():
        print("\nGPU")
        device_gpu = torch.device('cuda:0')
        print("Torch:")
        measure_dataloader(m.DataLoader.Torch, batch_size, device_gpu, dataset, sample_size)
        print("SequentialHDF5:")
        measure_dataloader(m.DataLoader.SequentialHDF5, batch_size, device_gpu, dataset, sample_size)
        print("ParallelHDF5:")
        measure_dataloader(m.DataLoader.ParallelHDF5, batch_size, device_gpu, dataset, sample_size)

if __name__ == "__main__":
    main()
