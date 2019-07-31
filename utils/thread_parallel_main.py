from main import *


def thread_parallel_main(argv):
    setRandomSeeds()

    dataset_path = '../datasets/'

    # Default Parameters
    datasetType = DataSet.MNIST
    modelType = Model.LeNet5Updated
    dataLoaderType = DataLoader.ParallelHDF5
    trainingType = Training.Sequential
    node_count = 1
    deviceType = Device.GPU

    max_epochs = 200
    max_epochs_without_improvement = 10
    learning_rate = 0.001
    momentum = 0.9
    batch_size = 64

    keepResults = True
    printProgress = False

    # Define command line options and help text
    short_opts = "hd:m:l:p:e:b:"
    long_opts = ["help", "data=", "model=", "net=", "dl=", "dataloader=", "node_count=", "epochs=",
                 "earlystopping=", "learningrate=", "momentum=", "bs=", "batchsize=", "discard-results",
                 "print-progress", "device="]

    help_text = """Train a neural network on an image classification problem (MNIST or CIFAR10) and collect
running times, errors and losses over time. Results are stored in 'outputs/'.
Expects the selected dataset as HDF5 file in '../datasets/'. Use createDatasets.py to create those input files.


Options:
    -h --help               Show this screen.

    -d --data               Select data set to train on
                            values: MNIST, CIFAR10 [default:MNIST]

    -m --model --net        Select model to train, values:
                            PyTorchTutorialNet, LeNet5, LeNet5Updated, AlexNet, AlexNetPool
                            [default:LeNet5Updated]

    -l --dl --dataloader    Select the data loader, values:
                            SequentialHDF5, ParallelHDF5, Torch [default:ParallelHDF5]

    -p --node_count         Number of threads [default:1]

    -b --bs --batchsize     Total batch size (over all processes) [default:64]

    -e --epochs             Maximum number of training epochs [default:200]

    --device                Switch between CPU and GPU, values: CPU, GPU [default: GPU]

    --earlystopping         Maximum number of epochs without improvement
                            before the training is stopped [default:10]

    --learningrate          Learning rate [default:0.001]

    --momentum              Training momentum [default:0.9]

    --discard-results       Do not store the training results in output/

    --print-progress        Print a progress bar and a short summary for each training epoch"""

    # Parse command line arguments
    try:
        opts, args = getopt.getopt(argv, short_opts, long_opts)
    except getopt.GetoptError:
        print(help_text)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_text)
            sys.exit()
        elif opt in ("-d", "--data"):
            datasetType = DataSet[arg]
        elif opt in ("-m", "--model", "--net"):
            modelType = Model[arg]
        elif opt in ("-l", "--dl", "--dataloader"):
            dataLoaderType = DataLoader[arg]
        elif opt in ("-p", "--node_count"):
            node_count = int(arg)
        elif opt in ("-b", "--bs", "--batchsize"):
            batch_size = int(arg)
        elif opt in ("-e", "--epochs"):
            max_epochs = int(arg)
        elif opt in ("--device"):
            deviceType = Device[arg]
        elif opt in ("--earlystopping"):
            max_epochs_without_improvement = int(arg)
        elif opt in ("--learningrate"):
            learning_rate = float(arg)
        elif opt in ("--momentum"):
            momentum = float(arg)
        elif opt in ("--discard-results"):
            keepResults = False
        elif opt in ("--print-progress"):
            printProgress = True

    torch.set_num_threads(node_count)

    config = {
        "dataset": datasetType.name,
        "dataLoader": dataLoaderType.name,
        "model": modelType.name,
        "training": "TorchThreadParallel",
        "max_epochs": max_epochs,
        "max_epochs_without_improvement": max_epochs_without_improvement,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "batch_size": batch_size,
        "node_count": node_count,
        "device": deviceType.name}
    metadata = "\n# " + str(config)

    print('Configuration:')
    for opt, value in config.items():
        print("  {:<31}  {}".format(opt + ':', value))
    print('')

    # Assign each process a device
    if deviceType == Device.GPU and torch.cuda.device_count() < 1:
        print("Error: selected GPU as device but no GPU devices available")
        return

    cpu_device = torch.device('cpu')
    gpu_device = torch.device('cuda', 0)
    device = cpu_device if deviceType == Device.CPU else gpu_device

    trainObj = train(datasetType, dataset_path, modelType, dataLoaderType, trainingType,
                     max_epochs, max_epochs_without_improvement, learning_rate, momentum,
                     batch_size, device, printProgress)

    now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    config["date"] = now
    unique_id = uuid.uuid4()
    trainObj.saveResults("outputs/results__" + now + "__" + str(unique_id), comment=metadata, config=config)


if __name__ == "__main__":
    thread_parallel_main(sys.argv[1:])
