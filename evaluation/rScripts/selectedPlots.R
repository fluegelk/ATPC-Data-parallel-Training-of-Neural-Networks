library(ggplot2)
library(reshape2)
library(RColorBrewer)
Sys.setenv(LANG = "en")
source("utils.R")
source("plots.R")

### -------- Configs --------
outpath <<- "../diagrams-manual/"
mkdir(outpath)
inpath <<- "../results/ForHLRII-CPU/"
loadDataDir(path=inpath)
inpath <<- "../results/LSDF-CPU/"
loadDataDir(path=inpath)
inpath <<- "../results/LSDF-GPU/"
loadDataDir(path=inpath)

rename_model_and_machine <- function(epoch_df) {
    epoch_df["model_renamed"] <- ifelse(epoch_df$model == "PyTorchTutorialNet", "TorchNet", as.character(epoch_df$model))
    epoch_df$model_renamed <- as.factor(epoch_df$model_renamed)
    epoch_df["machine_renamed"] <- ifelse(epoch_df$machine == "ForHLR2", "ForHLR II", as.character(epoch_df$machine))
    epoch_df$machine_renamed <- as.factor(epoch_df$machine_renamed)
    return(epoch_df)
}

add_ForHLR2_Sequential <- function(epoch_df) {
    ForHLR2_Sequential <- subset(epoch_df, machine == "ForHLR2" & nodeCount == 1 & training == "AllReduce")
    ForHLR2_Sequential$training <- "Sequential"
    return(rbind(epoch_df, ForHLR2_Sequential))
}

add_training_node_count <- function(epoch_df) {
    epoch_df["training_node_count"] <- as.factor(paste(epoch_df$training, epoch_df$nodeCount))
    levels <- c("Sequential 1", "AllReduce 1", "AllReduce 2", "AllReduce 4", "AllReduce 8",
        "AllReduce 16", "Threadparallel 4", "Threadparallel 16")
    epoch_df <- within(epoch_df, training_node_count <- factor(training_node_count, levels=levels))
    return(epoch_df)
}

training_node_count_labels <- c("Sequential 1" = "Seq", "AllReduce 1" = "p1", "AllReduce 2" = "p2",
    "AllReduce 4" = "p4", "AllReduce 8" = "p8", "AllReduce 16" = "p16",
    "Threadparallel 4" = "t4", "Threadparallel 16" = "t16")

data <- epochData
data <- rename_model_and_machine(data)
data <- add_training_node_count(data)

data_with_ForHLR2_Sequential <- epochData
data_with_ForHLR2_Sequential <- add_ForHLR2_Sequential(data_with_ForHLR2_Sequential)
data_with_ForHLR2_Sequential <- rename_model_and_machine(data_with_ForHLR2_Sequential)
data_with_ForHLR2_Sequential <- add_training_node_count(data_with_ForHLR2_Sequential)

data_id <- c(data_id, "machine_renamed", "model_renamed")

loss_threshold <- data.frame("loss_threshold" = c(0.5,1.5), "dataset" = c("MNIST", "CIFAR10"))
threshold_epochs <- determine_threshold_epoch(data_with_ForHLR2_Sequential, loss_threshold)

update_geom_defaults("line", list(size = 1.5))
update_geom_defaults("point", list(size = 3))

plot_theme <- function(...) {
    list(
        theme_classic(...),
        theme(plot.title = element_text(size=20, margin=margin(t=10, b=10)), legend.key=element_blank(), text = element_text(size=20)),
        guides(colour = guide_legend(ncol = 1))
    )
}

### -------- Plot Titles --------
speedup1_title <- function(machine, device, dataset) {
    return(paste("Speedup by Batch Size\non", machine, device, "with", dataset, "and LeNet5"))
}
speedup2_title <- function(machine, device) {
    return(paste("Speedup with Batch Size 256 on", machine, device))
}
commtime1_title <- function(dataset) {
    return(paste("Mean Epoch Time with Batch Size 256, LeNet5 and", dataset))
}
commtime2_title <- function(machine, device, nodeCount, dataset) {
    return(paste("Mean Epoch Time by Batch Size\non", machine, device, "with p =", paste(nodeCount, ",", sep=""), "LeNet5 and", dataset))
}
loss1_title <- function(dataset) {
    return(paste("Validation Loss by Epoch and Batch Sizeon GPU\nwith Sequential Training, LeNet5 and", dataset))
}
loss2_title <- function(dataset) {
    return(paste("Validation Loss by Time and Batch Size on GPU\nwith Sequential Training, LeNet5 and", dataset))
}
loss3_title <- function(dataset) {
    return(paste("Validation Loss by Time on GPU\nwith Batch Size 256, LeNet5 and", dataset))
}
loss4_title <- function(dataset) {
    return(paste("Validation Loss by Time on GPU\nwith Mini-Batch Size 32, LeNet5 and", dataset))
}
loss5_title <- function(machine, device, dataset) {
    return(paste("Validation Loss by Time with Batch Size 256\non", machine, device, "with", dataset))
}

### -------- Plot File Names --------
speedup1_filename <- function(machine, device, dataset) {
    return(paste("speedup-", machine, device, dataset, sep="-"))
}
speedup2_filename <- function(machine, device) {
    return(paste("speedup-bs-256-", machine, device, sep="-"))
}
commtime1_filename <- function(dataset) {
    return(paste("epoch-time-bs-256-", dataset, sep="-"))
}
commtime2_filename <- function(machine, device, dataset) {
    return(paste("epoch-time-", machine, device, dataset, sep="-"))
}
loss1_filename <- function(dataset) {
    return(paste("validation-loss-by-epochs-", dataset, sep="-"))
}
loss2_filename <- function(dataset) {
    return(paste("validation-loss-by-time-and-bs-", dataset, sep="-"))
}
loss3_filename <- function(dataset) {
    return(paste("validation-loss-by-time-bs-256-", dataset, sep="-"))
}
loss4_filename <- function(dataset) {
    return(paste("validation-loss-by-time-mbs-32-", dataset))
}
loss5_filename <- function(machine, device, dataset) {
    return(paste("validation-loss-by-time-and-hw-", machine, device, dataset, sep="-"))
}

### -------- Plot & Save Functions --------
plot_speedup1 <- function(data, var_machine, var_device, var_dataset, path) {
    data <- subset(data, model == "LeNet5")
    data <- subset(data, machine_renamed == var_machine & device == var_device & dataset == var_dataset)

    maxThreads <- max(data$nodeCount)
    data <- prepare_speedup_data(data, threshold_epochs)
    data <- subset(data, variable %in% c("total_speedup", "total_efficiency"))

    title <- speedup1_title(var_machine, var_device, var_dataset)
    filename <- speedup1_filename(var_machine, var_device, var_dataset)
    plot <- speedup_plot_helper(data, maxThreads, title, "batch_size", "Batch Size")
    save_plot(plot, filename, path=path)
}
plot_speedup2 <- function(data, var_machine, var_device, path) {
    data <- subset(data, batch_size == 256)
    data <- subset(data, machine_renamed == var_machine & device == var_device)

    maxThreads <- max(data$nodeCount)
    data <- prepare_speedup_data(data, threshold_epochs)
    data <- subset(data, variable %in% c("total_speedup", "total_efficiency"))

    title <- speedup2_title(var_machine, var_device)
    filename <- speedup2_filename(var_machine, var_device)
    data["model_dataset"] <- paste(data$model_renamed, data$dataset)
    plot <- speedup_plot_helper(data, maxThreads, title, "model_dataset", "Model and Dataset")
    save_plot(plot, filename, path=path, width=10)
}
plot_commtime1 <- function(data, var_dataset, path) {
    data <- subset(data, model == "LeNet5" & batch_size == 256)
    data <- subset(data, dataset == var_dataset)
    data <- aggregate_comp_comm_time(data, c(data_id, "training_node_count"), mean)

    title <- commtime1_title(var_dataset)
    filename <- commtime1_filename(var_dataset)

    facet <- facet_wrap(machine_renamed~device, nrow = 1, scales="free")
    plot <- comp_comm_time_plot(data, title, "training_node_count", "Training", facet=facet) +
                    scale_x_discrete(labels=training_node_count_labels)
    save_plot(plot, filename, path=path, width=12)
}
plot_commtime2 <- function(data, var_machine, var_device, var_dataset, path) {
    data <- subset(data, model == "LeNet5")
    data <- subset(data, machine_renamed == var_machine & device == var_device & dataset == var_dataset)
    var_nodeCount <- max(data$nodeCount)
    data <- subset(data, nodeCount == var_nodeCount)
    id_vars <- c(data_id[data_id != "mini_batch_size"], "training_node_count")
    data <- aggregate_comp_comm_time(data, id_vars, mean)

    title <- commtime2_title(var_machine, var_device, var_nodeCount, var_dataset)
    filename <- commtime2_filename(var_machine, var_device, var_dataset)

    plot <- comp_comm_time_plot(data, title, "batch_size", "Batch Size", facet=NULL)
    save_plot(plot, filename, path=path)
}
plot_loss1 <- function(data, var_dataset, path) {
    data <- subset(data, model == "LeNet5" & training == "Sequential" & device == "GPU" & machine == "LSDF")
    data <- subset(data, dataset == var_dataset)
    title <- loss1_title(var_dataset)
    filename <- loss1_filename(var_dataset)
    plot <- loss_plot(data, title, color="batch_size", color_label="Batch Size", x="epoch", x_label="Epochs")
    save_plot(plot, filename, path=path)
}
plot_loss2 <- function(data, var_dataset, path) {
    data <- subset(data, model == "LeNet5" & training == "Sequential" & device == "GPU" & machine == "LSDF")
    data <- subset(data, dataset == var_dataset)
    title <- loss2_title(var_dataset)
    filename <- loss2_filename(var_dataset)
    plot <- loss_plot(data, title, color="batch_size", color_label="Batch Size",
        x="summedTotalTime", x_label="Training Time [s]")
    save_plot(plot, filename, path=path)
}
plot_loss3 <- function(data, var_dataset, path) {
    data <- subset(data, model == "LeNet5" & batch_size == 256 & device == "GPU" & machine == "LSDF")
    data <- subset(data, dataset == var_dataset)
    title <- loss3_title(var_dataset)
    filename <- loss3_filename(var_dataset)
    plot <- loss_plot(data, title, color="training_node_count", color_label="Training",
        x="summedTotalTime", x_label="Training Time [s]")
    save_plot(plot, filename, path=path)
}
plot_loss4 <- function(data, var_dataset, path) {
    data <- subset(data, model == "LeNet5" & mini_batch_size == 32 & device == "GPU" & machine == "LSDF")
    data <- subset(data, dataset == var_dataset)
    title <- loss4_title(var_dataset)
    filename <- loss4_filename(var_dataset)
    plot <- loss_plot(data, title, color="training_node_count", color_label="Training",
        x="summedTotalTime", x_label="Training Time [s]")
    save_plot(plot, filename, path=path)
}
plot_loss5 <- function(data, var_machine, var_device, var_dataset, path) {
    data <- subset(data, batch_size == 256)
    data <- subset(data, machine_renamed == var_machine & device == var_device & dataset == var_dataset)
    data <- subset(data, training == "Sequential" | nodeCount == max(data$nodeCount))

    title <- loss5_title(var_machine, var_device, var_dataset)
    filename <- loss5_filename(var_machine, var_device, var_dataset)
    plot <- loss_plot(data, title, color="model_renamed", color_label="Model",
        shape="training_node_count", shape_label="Training",
        x="summedTotalTime", x_label="Training Time [s]")
    save_plot(plot, filename, path=path)
}

path <- outpath
filteredEpochData <- filter_epochs_by_loss_threshold(data, threshold_epochs)
filteredEpochData <- subset(filteredEpochData, epoch != 0)

plot_speedup1(data_with_ForHLR2_Sequential, "LSDF", "CPU", "MNIST", path)
plot_speedup1(data_with_ForHLR2_Sequential, "LSDF", "CPU", "CIFAR10", path)
plot_speedup1(data_with_ForHLR2_Sequential, "LSDF", "GPU", "MNIST", path)
plot_speedup1(data_with_ForHLR2_Sequential, "LSDF", "GPU", "CIFAR10", path)
plot_speedup1(data_with_ForHLR2_Sequential, "ForHLR II", "CPU", "MNIST", path)
plot_speedup1(data_with_ForHLR2_Sequential, "ForHLR II", "CPU", "CIFAR10", path)

plot_speedup2(data_with_ForHLR2_Sequential, "LSDF", "CPU", path)
plot_speedup2(data_with_ForHLR2_Sequential, "LSDF", "GPU", path)
plot_speedup2(data_with_ForHLR2_Sequential, "ForHLR II", "CPU", path)

plot_commtime1(filteredEpochData, "MNIST", path)
plot_commtime1(filteredEpochData, "CIFAR10", path)

plot_commtime2(filteredEpochData, "LSDF", "CPU", "MNIST", path)
plot_commtime2(filteredEpochData, "LSDF", "CPU", "CIFAR10", path)
plot_commtime2(filteredEpochData, "LSDF", "GPU", "MNIST", path)
plot_commtime2(filteredEpochData, "LSDF", "GPU", "CIFAR10", path)
plot_commtime2(filteredEpochData, "ForHLR II", "CPU", "MNIST", path)
plot_commtime2(filteredEpochData, "ForHLR II", "CPU", "CIFAR10", path)

plot_loss1(data_with_ForHLR2_Sequential, "MNIST", path)
plot_loss1(data_with_ForHLR2_Sequential, "CIFAR10", path)

plot_loss2(data_with_ForHLR2_Sequential, "MNIST", path)
plot_loss2(data_with_ForHLR2_Sequential, "CIFAR10", path)

plot_loss3(data_with_ForHLR2_Sequential, "MNIST", path)
plot_loss3(data_with_ForHLR2_Sequential, "CIFAR10", path)

plot_loss4(data_with_ForHLR2_Sequential, "MNIST", path)
plot_loss4(data_with_ForHLR2_Sequential, "CIFAR10", path)

plot_loss5(data_with_ForHLR2_Sequential, "LSDF", "CPU", "MNIST", path)
plot_loss5(data_with_ForHLR2_Sequential, "LSDF", "CPU", "CIFAR10", path)
plot_loss5(data_with_ForHLR2_Sequential, "LSDF", "GPU", "MNIST", path)
plot_loss5(data_with_ForHLR2_Sequential, "LSDF", "GPU", "CIFAR10", path)
plot_loss5(data_with_ForHLR2_Sequential, "ForHLR II", "CPU", "MNIST", path)
plot_loss5(data_with_ForHLR2_Sequential, "ForHLR II", "CPU", "CIFAR10", path)