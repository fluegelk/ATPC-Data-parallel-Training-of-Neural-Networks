library(ggplot2)
library(reshape2)
library(RColorBrewer)
library(extrafont)
library(cowplot)
Sys.setenv(LANG = "en")
source("utils.R")
source("plots.R")

### -------- Configs --------
outpath <<- "../diagrams-paper/"
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

update_geom_defaults("line", list(size = 0.5))
update_geom_defaults("point", list(size = 0.5))
wide_legend <- guide_legend(
                 keywidth=0.25,
                 keyheight=0.1,
                 default.unit="inch",
                 ncol=1)

default_plot_units <- "in"
default_plot_width <- 2.5
double_col_plot_width <- 7
default_plot_height <- 1.5

loadfonts(quiet=TRUE)
default_text_size <- 8
default_font_family="Times New Roman"

theme_set(theme_cowplot(font_size=default_text_size, font_family = default_font_family))

plot_theme <- function(...) {
    default_text <- element_text(size=default_text_size, margin=margin(0,0,0,0), family=default_font_family)
    title_text <- element_text(size=10, family=default_font_family)
    list(
        theme_classic(...),
        theme(legend.key=element_blank(), text = default_text,
            legend.text = default_text,
            axis.text = default_text,
            legend.spacing = unit(0.1, 'in'),
            legend.key.size=unit(0.1, "in"),
            legend.box.spacing=unit(0.1, "in"),
            legend.margin=margin(0, 0, 0, 0, unit = "pt"),
            legend.position="bottom",
            legend.direction="horizontal",
            strip.background = element_blank(),
            strip.text = title_text,
            plot.title = title_text
            )
    )
}

### -------- Plot File Names --------
speedup_filename <- function(machine, device, dataset) {
    return(paste("speedup-", machine, device, dataset, sep="-"))
}
speedup2_filename <- function(batch_size) {
    return(paste("speedup-all--bs", batch_size, sep="-"))
}
commtime_hardware_filename <- function(dataset) {
    return(paste("epoch-time-bs-256-", dataset, sep="-"))
}
commtime_batchsize_filename <- function(machine, device, dataset) {
    return(paste("epoch-time-", machine, device, dataset, sep="-"))
}
loss1_filename <- function(dataset) {
    return(paste("validation-loss-by-epochs-", dataset, sep="-"))
}
loss2_filename <- function(dataset) {
    return(paste("validation-loss-by-time-and-bs-", dataset, sep="-"))
}
loss3_filename <- function(batch_size) {
    return(paste("validation-loss-by-model-and-dataset--bs", batch_size, sep="-"))
}

### -------- Plot & Save Functions --------
print_speedups <- function(data, var_dataset="MNIST", var_model="LeNet5") {
    data <- subset(data, dataset == var_dataset & model == var_model)
    data <- prepare_speedup_data(data, threshold_epochs)

    data$nodeCount <- ifelse(data$device == "CPU" & data$nodeCount == 4, 0, data$nodeCount)
    data <- subset(data, batch_size %in% c(16,512) & nodeCount %in% c(4,16) & variable %in% c("mean_speedup", "total_speedup"))
    data <- data[c("batch_size", "nodeCount", "device", "machine_renamed", "variable", "value")]


    print(data)
}


plot_speedup <- function(data, var_machine, var_device, var_dataset="MNIST", var_model="LeNet5", path) {
    data <- subset(data, machine_renamed == var_machine & device == var_device & dataset == var_dataset & model == var_model)
    data <- prepare_speedup_data(data, threshold_epochs)
    maxThreads <- max(data$nodeCount)

    mean_vars <- c("mean_speedup", "mean_efficiency")
    total_vars <- c("total_speedup", "total_efficiency")

    plot_helper <- function(data, title) {
        speedup_plot_helper(data, maxThreads, title, "batch_size", "Batch Size") +
            scale_y_continuous(breaks=c(1,2,3,4), sec.axis = sec_axis(~./maxThreads, name = "Efficiency")) +
            scale_x_continuous(breaks=c(1,2,3,4))
    }

    total_speedup <- plot_helper(subset(data, variable %in% total_vars), "Total Training Speedup") +
        theme(legend.position="right", legend.direction="vertical")

    mean_speedup <- plot_helper(subset(data, variable %in% mean_vars), "Speedup per Epoch")+
        theme(legend.position="none")

    plot <- plot_grid(mean_speedup, total_speedup, labels = "AUTO", rel_widths = c(1, 1.5))

    filename <- speedup_filename(var_machine, var_device, var_dataset)
    save_plot(plot, filename, path=path, height=default_plot_height*1.5, width=double_col_plot_width*0.85)
}

plot_speedup2 <- function(data, var_machine="LSDF", var_device="GPU", var_batch_size=256, path) {
    data <- subset(data, machine_renamed == var_machine & device == var_device & batch_size == var_batch_size)
    data <- prepare_speedup_data(data, threshold_epochs)
    data["model_dataset"] <- paste(data$model_renamed, data$dataset)
    maxThreads <- max(data$nodeCount)

    mean_vars <- c("mean_speedup", "mean_efficiency")
    total_vars <- c("total_speedup", "total_efficiency")

    plot_helper <- function(data, title) {
        speedup_plot_helper(data, maxThreads, title, "model_dataset", "Model and Dataset", gradient=FALSE) +
            scale_y_continuous(breaks=c(1,2,3,4), sec.axis = sec_axis(~./maxThreads, name = "Efficiency")) +
            scale_x_continuous(breaks=c(1,2,3,4))
    }

    mean_speedup <- plot_helper(subset(data, variable %in% mean_vars), NULL)+
        theme(legend.position="bottom", legend.direction="vertical")+
        guides(color=guide_legend(ncol=1))

    filename <- speedup2_filename(var_batch_size)
    save_plot(mean_speedup, filename, path=path, height=default_plot_height*2)
}

plot_commtime_hardware <- function(data, var_machine, var_device, var_dataset="MNIST", var_model="LeNet5", var_batch_size=256, path) {
    data <- subset(data, dataset == var_dataset & model == var_model & batch_size == var_batch_size)
    data <- subset(data, training != "Sequential")
    data <- aggregate_comp_comm_time(data, c(data_id, "training_node_count"), mean)
    data["machine_device"] <- paste(data$machine_renamed, data$device)

    plot_helper <- function(data, title) {
        comp_comm_time_plot(data, title, "nodeCount", "# Processes", facet=NULL) +
                    scale_x_discrete(labels=pow2s)
    }

    lsdf_cpu <- plot_helper(subset(data, machine_device == "LSDF CPU"), "LSDF CPU") +
        theme(legend.position = c(0.7, 0.9)) +
        guides(fill=guide_legend(ncol=1))
    lsdf_gpu <- plot_helper(subset(data, machine_device == "LSDF GPU"), "LSDF GPU") +
        theme(legend.position="none")
    forhlr <- plot_helper(subset(data, machine_device == "ForHLR II CPU"), "ForHLR II CPU") +
        theme(legend.position="none")

    plot <- plot_grid(lsdf_cpu, lsdf_gpu, forhlr, labels = "AUTO", nrow=1)

    filename <- commtime_hardware_filename(var_dataset)
    save_plot(plot, filename, path=path, height=default_plot_height*1.2, width=double_col_plot_width)
}

plot_commtime_batchsize <- function(data, var_machine, var_device, var_dataset="MNIST", var_model="LeNet5", path) {
    data <- subset(data, machine_renamed == var_machine & device == var_device & dataset == var_dataset & model == var_model)
    var_nodeCount <- max(data$nodeCount)

    data <- subset(data, nodeCount == var_nodeCount)
    id_vars <- c(data_id[data_id != "mini_batch_size"], "training_node_count")
    data <- aggregate_comp_comm_time(data, id_vars, mean)

    plot <- comp_comm_time_plot(data, NULL, "batch_size", "Batch Size", facet=NULL)
    plot <- plot +
        theme(legend.position = c(0.7, 0.8)) +
        guides(fill=guide_legend(ncol=1))
    filename <- commtime_batchsize_filename(var_machine, var_device, var_dataset)
    save_plot(plot, filename, path=path)
}

MNIST_Loss_Legend <- list(
    theme(legend.position = c(0.8, 0.7), legend.direction="vertical"),
    guides(color=guide_legend(ncol=2))
)

CIFAR_Loss_Legend <- list(
    theme(legend.position = c(0.5, 0.2), legend.direction="horizontal"),
    guides(color=guide_legend(ncol=3))
)

plot_sequential_batch_size_loss <- function(data, var_machine, var_device, var_dataset, var_model, path, var_x, x_label, filename) {
    data <- subset(data, machine_renamed == var_machine & device == var_device & dataset == var_dataset & model == var_model)
    data <- subset(data, training == "Sequential")

    plot <- loss_plot(data, NULL, color="batch_size", color_label="Batch Size", x=var_x, x_label=x_label)
    if (var_dataset == "MNIST") { plot <- plot + MNIST_Loss_Legend }
    else { plot <- plot + CIFAR_Loss_Legend }

    save_plot(plot, filename, path=path)
}

plot_loss1 <- function(data, var_machine="LSDF", var_device="GPU", var_dataset="MNIST", var_model="LeNet5", path) {
    filename <- loss1_filename(var_dataset)
    plot_sequential_batch_size_loss(data, var_machine, var_device, var_dataset, var_model, path, "epoch", "Epochs", filename)
}

plot_loss2 <- function(data, var_machine="LSDF", var_device="GPU", var_dataset="MNIST", var_model="LeNet5", path) {
    filename <- loss2_filename(var_dataset)
    plot_sequential_batch_size_loss(data, var_machine, var_device, var_dataset,
        var_model, path, "summedTotalTime", "Training Time [s]", filename)
}

plot_loss3 <- function(data, var_machine="LSDF", var_device="GPU", var_batch_size=256, path) {
    data <- subset(data, machine_renamed == var_machine & device == var_device & batch_size == var_batch_size)
    data <- subset(data, training == "Sequential")

    plot <- loss_plot(data, NULL, color="model_renamed", color_label="Model",
        x="epoch", x_label="Epochs", gradient=FALSE) +
        theme(legend.position = "bottom", legend.direction="horizontal") +
        guides(color=guide_legend(nrow=1)) +
        facet_wrap(~dataset, scales="free_x")

    filename <- loss3_filename(var_batch_size)
    save_plot(plot, filename, path=path)
}

path <- outpath
filteredEpochData <- filter_epochs_by_loss_threshold(data, threshold_epochs)
filteredEpochData <- subset(filteredEpochData, epoch != 0)

print_speedups(data=data_with_ForHLR2_Sequential)
plot_speedup(data=data_with_ForHLR2_Sequential, var_machine="LSDF", var_device="GPU", path=path)
plot_speedup2(data=data_with_ForHLR2_Sequential, path=path)

plot_commtime_hardware(data=filteredEpochData, path=path)
plot_commtime_batchsize(data=filteredEpochData, var_machine="LSDF", var_device="GPU", path=path)

plot_loss1(data=data_with_ForHLR2_Sequential, path=path)
plot_loss2(data=data_with_ForHLR2_Sequential, path=path)
plot_loss3(data=data_with_ForHLR2_Sequential, path=path)

plot_loss1(data=data_with_ForHLR2_Sequential, var_dataset="CIFAR10", path=path)
plot_loss2(data=data_with_ForHLR2_Sequential, var_dataset="CIFAR10", path=path)