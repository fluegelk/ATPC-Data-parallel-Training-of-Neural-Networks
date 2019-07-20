library(ggplot2)
library(reshape2)
library(RColorBrewer)
Sys.setenv(LANG = "en")
source("utils.R")
source("plots.R")

### -------- Load Data --------

loadLocal <- function() {
    # Sequential & AllReduce, bs {32, 64, 128}, p {1, 2}
    loadDataDir(path=inpath)
}

load_lsdf_28_131_seq <- function() {
    # LeNet5Updated, ParallelHDF5, p = 1, Sequential & AllReduce, bs {16, 64, 256}, CIFAR10 & MNIST
    outpath <<- "../lsdf-28-131-seq/"
    inpath <<- "../../implementation/outputs/lsdf-28-131__06-27/"
    loadDataDir(path=inpath)
}

load_lsdf_28_131_par_cpu <- function() {
    # LeNet5Updated, ParallelHDF5
    outpath <<- "../lsdf-28-131-par-cpu/"
    inpath <<- "../../implementation/outputs/lsdf-28-131__07-01/"
    loadDataDir(path=inpath)
}

load_final_data <- function() {
    outpath <<- "../diagrams/"
    # inpath <<- "../results/ForHLRII-CPU/"
    # loadDataDir(path=inpath)
    inpath <<- "../results/LSDF-CPU/"
    loadDataDir(path=inpath)
    inpath <<- "../results/LSDF-GPU/"
    loadDataDir(path=inpath)
}

### -------- Configs --------
# loadLocal()
# load_lsdf_28_131_seq()
# load_lsdf_28_131_par_cpu()
load_final_data()

# TODO: select threshold values!
loss_threshold <- data.frame("loss_threshold" = c(0.1,0.5), "dataset" = c("MNIST", "CIFAR10"))
threshold_epochs <- determine_threshold_epoch(epochData, loss_threshold)
print(threshold_epochs[c("dataset", "model", "training", "machine", "device", "nodeCount", "batch_size", "threshold_epoch")])

data <- epochData
data["training_node_count"] <- as.factor(paste(data$training, data$nodeCount))
levels <- c("Sequential 1", "AllReduce 1", "AllReduce 2", "AllReduce 4", "AllReduce 8",
    "AllReduce 16", "Threadparallel 4", "Threadparallel 16")
data <- within(data, training_node_count <- factor(training_node_count, levels=levels))

### -------- Plot and Save --------

selected_batch_size <- 256
selected_mini_batch_size <- 16

# data <- filter_epochs_by_loss_threshold(epochData, threshold_epochs) # TODO

plot_speedup <- function(data, file_extension, title_extension, threshold_epochs) {
    plots <- speedupEfficiencyPlots(data, threshold_epochs)
    save_plot(plots[[1]], paste("meanSpeedup", file_extension, sep="-"))
    save_plot(plots[[2]], paste("totalSpeedup", file_extension, sep="-"))
}

plot_with_batch_size <- function(plot_helper, data, file_extension, title_extension, selected_batch_size, ...) {
    comm_data <- subset(data, batch_size == selected_batch_size)
    title_extension <- paste("With Batch Size ", selected_batch_size, "\n", title_extension, sep="")
    file_extension <- paste("bs", selected_batch_size, file_extension, sep="-")
    plot_helper(comm_data, title_extension, file_extension, ...)
}

plot_with_mini_batch_size <- function(plot_helper, data, file_extension, title_extension, selected_mini_batch_size, ...) {
    comm_data <- subset(data, mini_batch_size == selected_mini_batch_size)
    title_extension <- paste("With Mini-Batch Size ", selected_mini_batch_size, "\n", title_extension, sep="")
    file_extension <- paste("mbs", selected_mini_batch_size, file_extension, sep="-")
    plot_helper(comm_data, title_extension, file_extension, ...)
}

plot_for_all_trainings_by_var <- function(plot_helper, data, path, file_extension, title_extension, var, var_label, ...) {
    mkdir(path)
    for (current_training_node_count in unique(data$training_node_count)) {
        comm_data <- subset(data, training_node_count == current_training_node_count)
        cur_title_extension <- paste("With Training", current_training_node_count, "\nby", var_label, title_extension)
        cur_file_extension <- paste("training", current_training_node_count, "by", var, file_extension, sep="-")
        plot_helper(comm_data, cur_title_extension, path, cur_file_extension, var, var_label, ...)
    }
}

for (current_dataset in unique(data$dataset)) {
    for (current_machine in unique(data$machine)) {
        current_data <- subset(data, dataset == current_dataset & machine == current_machine)
        current_file_extension <- paste(current_dataset, current_machine, sep="-")
        current_title_extension <- paste("(", current_dataset, ", ", current_machine, ")", sep="")

        # Speedup plots
        # print(nrow(subset(current_data, training == "Sequential")))
        print(nrow(current_data))
        print(current_title_extension)
        plot_speedup(current_data, current_file_extension, current_title_extension, threshold_epochs)

        # Communication time plots
        filtered_data <- filter_epochs_by_loss_threshold(current_data, threshold_epochs)
        ## With batch size == selected_batch_size
        plot_with_batch_size(comm_by_training_helper, filtered_data, current_file_extension,
            current_title_extension, selected_batch_size)
        ## With mini_batch size == selected_mini_batch_size
        plot_with_mini_batch_size(comm_by_training_helper, filtered_data, current_file_extension,
            current_title_extension, selected_mini_batch_size)
        ## for each training with different (mini) batch sizes
        path <- paste(outpath, "comm_time_by_batch_size/", sep="")
        plot_for_all_trainings_by_var(comm_by_var_helper, filtered_data, path, current_file_extension,
            current_title_extension, "batch_size", "Batch Size")
        path <- paste(outpath, "comm_time_by_mini_batch_size/", sep="")
        plot_for_all_trainings_by_var(comm_by_var_helper, filtered_data, path, current_file_extension,
            current_title_extension, "mini_batch_size", "Mini-Batch Size")

        # Validation loss plots
        ## With batch size == selected_batch_size
        plot_with_batch_size(loss_by_training_helper, current_data, current_file_extension,
            current_title_extension, selected_batch_size, facet=model_device_facet(scales="free"))
        ## With mini_batch size == selected_mini_batch_size
        plot_with_mini_batch_size(loss_by_training_helper, current_data, current_file_extension,
            current_title_extension, selected_mini_batch_size, facet=model_device_facet(scales="free"))
        ## for each training with different (mini) batch sizes
        path <- paste(outpath, "loss_by_batch_size/", sep="")
        plot_for_all_trainings_by_var(loss_by_var_helper, current_data, path, current_file_extension,
            current_title_extension, "batch_size", "Batch Size", facet=model_device_facet(scales="free"))
        path <- paste(outpath, "loss_by_batch_size/", sep="")
        plot_for_all_trainings_by_var(loss_by_var_helper, current_data, path, current_file_extension,
            current_title_extension, "mini_batch_size", "Mini-Batch Size", facet=model_device_facet(scales="free"))
    }
}

for (current_device in unique(data$device)) {
    for (current_machine in unique(data$machine)) {
        current_data <- subset(data, device == current_device & machine == current_machine)
        current_file_extension <- paste(current_machine, current_device, sep="-")
        current_title_extension <- paste("(", current_machine, ", ", current_device, ")", sep="")

        # Validation loss plots
        ## With batch size == selected_batch_size
        plot_with_batch_size(loss_by_training_helper, current_data, current_file_extension,
            current_title_extension, selected_batch_size, facet=model_dataset_facet(scales="free"))
        ## With mini_batch size == selected_mini_batch_size
        plot_with_mini_batch_size(loss_by_training_helper, current_data, current_file_extension,
            current_title_extension, selected_mini_batch_size, facet=model_dataset_facet(scales="free"))
        ## for each training with different (mini) batch sizes
        path <- paste(outpath, "loss_by_batch_size/", sep="")
        plot_for_all_trainings_by_var(loss_by_var_helper, current_data, path, current_file_extension,
            current_title_extension, "batch_size", "Batch Size", facet=model_dataset_facet(scales="free"))
        path <- paste(outpath, "loss_by_batch_size/", sep="")
        plot_for_all_trainings_by_var(loss_by_var_helper, current_data, path, current_file_extension,
            current_title_extension, "mini_batch_size", "Mini-Batch Size", facet=model_dataset_facet(scales="free"))
    }
}
