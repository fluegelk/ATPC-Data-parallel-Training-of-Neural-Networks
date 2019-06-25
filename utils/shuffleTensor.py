import torch
import torchvision
import numpy as np
import time


# Before: Tensor on device (cpu or gpu)
# After: Tensor on device but shuffled
#
# Quality: time to shuffle, time to iterate tensor
#
# Variants:
#
# Variant 1:
# - (copy from gpu to cpu if necessary)
# - convert to numpy
# - shuffle with numpy
# - convert to tensor
# - (copy to gpu if necessary)
#
# Variant 2:
# - generate random permutation of indices
# - create new view to tensor


def create_tensor(image_count=10000, device=torch.device('cpu')):
    return torch.rand(image_count, 32, 32, 3, device=device)


def iterate(tensor):
    """Iterate through tensor and set first channel of the pixel to a random number"""
    new_value = np.random.rand(1)[0]
    for image in tensor:
        image[0][0][0] = new_value


def numpy_shuffle(tensor):
    """Variant 1: shuffle with numpy, copy from gpu to cpu and back if tensor is on gpu"""
    start_shuffle = time.time()
    device = tensor.device
    if tensor.is_cuda:                    # copy from gpu to cpu if necessary
        tensor = tensor.cpu()
    tensor = tensor.numpy()               # convert to numpy
    np.random.shuffle(tensor)             # shuffle with numpy
    tensor = torch.from_numpy(tensor)     # convert to tensor
    tensor = tensor.to(device)            # copy back to original device if necessary
    end_shuffle = time.time()

    start_iteration = time.time()
    iterate(tensor)
    end_iteration = time.time()

    return end_shuffle - start_shuffle, end_iteration - start_iteration


def index_shuffle(tensor):
    """Variant 2: generate random permutation of indices and access data with view"""
    start_shuffle = time.time()
    index_permutation = torch.randperm(tensor.size()[0])  # generate random permutation of indices
    tensor = tensor[index_permutation]                    # create new view to tensor
    end_shuffle = time.time()

    start_iteration = time.time()
    iterate(tensor)
    end_iteration = time.time()

    return end_shuffle - start_shuffle, end_iteration - start_iteration


def measure_time(tensor, shuffle_func, sample_size, print_all=False):
    shuffle_time_sum = 0
    iteration_time_sum = 0
    msg = "    Shuffle: {:.4f}s\t     Iteration: {:.4f}s"
    for i in range(sample_size + 1):
        shuffle_time, iteration_time = shuffle_func(tensor)
        if i == 0:
            continue
        shuffle_time_sum += shuffle_time
        iteration_time_sum += iteration_time
        if print_all:
            print(msg.format(shuffle_time, iteration_time))
    msg = "Avg Shuffle: {:.4f}s\t Avg Iteration: {:.4f}s"
    print(msg.format(shuffle_time_sum / sample_size, iteration_time_sum / sample_size))


def iterate_dataloader(dataloader):
    new_value = np.random.rand(1)[0]
    for i, data in enumerate(self.trainloader, 0):
        images, labels = data
        for image in images:
            image[0][0][0] = new_value


def main():
    image_count = 60000
    sample_size = 5

    print("CPU")
    tensor_cpu = create_tensor(image_count=image_count)
    print("Numpy shuffle:")
    measure_time(tensor_cpu, numpy_shuffle, sample_size)
    print("Index shuffle:")
    measure_time(tensor_cpu, index_shuffle, sample_size)

    if torch.cuda.is_available():
        print("\nGPU")
        tensor_gpu = create_tensor(image_count=image_count, device=torch.device('cuda:0'))
        print("Numpy shuffle")
        measure_time(tensor_gpu, numpy_shuffle, sample_size)
        print("Index shuffle")
        measure_time(tensor_gpu, index_shuffle, sample_size)

if __name__ == "__main__":
    main()
