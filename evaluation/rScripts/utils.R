library(ggplot2)
library(reshape2)
library(RColorBrewer)
Sys.setenv(LANG = "en")

### -------- Loading Data from input files --------
inpath <- "../../implementation/outputs/"
epochData <- data.frame(epoch=double(),
                 totalTime=double(),
                 computationTime=double(),
                 communicationTime=double(),
                 validationError=double(),
                 validationLoss=double(),
                 trainingLoss=double(),
                 summedTotalTime=double(),
                 # from summary data
                 batch_size=integer(),
                 mini_batch_size=integer(),
                 nodeCount=integer(),
                 epochCount=integer(),
                 dataset=character(),
                 dataLoader=character(),
                 model=character(),
                 training=character(),
                 learning_rate=double(),
                 momentum=double(),
                 date=character(),
                 total_training_time=double(),
                 device=character(),
                 machine=character())

summaryData <- data.frame(batch_size=integer(),
                 mini_batch_size=integer(),
                 nodeCount=integer(),
                 epochCount=integer(),
                 totalTime=double(),
                 validationError=double(),
                 validationLoss=double(),
                 dataset=character(),
                 dataLoader=character(),
                 model=character(),
                 training=character(),
                 max_epochs=integer(),
                 max_epochs_without_improvement=integer(),
                 learning_rate=double(),
                 momentum=double(),
                 date=character(),
                 device=character(),
                 machine=character())

data_id <- c("batch_size", "mini_batch_size", "nodeCount", "epochCount", "dataset", "dataLoader", "model",
    "training", "device", "learning_rate", "momentum", "date", "machine")

loadData <- function(name) {
    summary  <- read.table(paste(inpath, name, "__summary", sep=""), header=TRUE)
    epochs <- read.table(paste(inpath, name, "__epochs", sep=""), header=TRUE)

    summary['mini_batch_size'] <- summary$batch_size / summary$nodeCount
    epochs["summedTotalTime"] <- cumsum(epochs$totalTime) # Reduce("+", epochs["totalTime"], accumulate = TRUE)

    epochs["batch_size"]          <- rep(summary$batch_size[[1]],nrow(epochs))
    epochs["mini_batch_size"]     <- rep(summary$mini_batch_size[[1]],nrow(epochs))
    epochs["nodeCount"]           <- rep(summary$nodeCount[[1]],nrow(epochs))
    epochs["epochCount"]           <- rep(summary$epochCount[[1]],nrow(epochs))
    epochs["dataset"]             <- rep(summary$dataset[[1]],nrow(epochs))
    epochs["dataLoader"]          <- rep(summary$dataLoader[[1]],nrow(epochs))
    epochs["model"]               <- rep(summary$model[[1]],nrow(epochs))
    epochs["training"]            <- rep(summary$training[[1]],nrow(epochs))
    epochs["learning_rate"]       <- rep(summary$learning_rate[[1]],nrow(epochs))
    epochs["momentum"]            <- rep(summary$momentum[[1]],nrow(epochs))
    epochs["date"]                <- rep(summary$date[[1]],nrow(epochs))
    epochs["total_training_time"] <- rep(summary$totalTime[[1]],nrow(epochs))
    epochs["device"]              <- rep(summary$device[[1]],nrow(epochs))
    epochs["machine"]              <- rep(summary$machine[[1]],nrow(epochs))

    # epochs$computationTime <- epochs$totalTime - epochs$communicationTime
    epochs$computationTime <- ifelse(epochs$training == "Sequential", epochs$totalTime, epochs$computationTime)

    epochData <<- rbind(epochData, epochs)
    summaryData <<- rbind(summaryData, summary)
}

loadDataDir <- function(path=".") {
    summary_files <- list.files(path = path, pattern = "*__summary",
        all.files = FALSE,
        full.names = FALSE,
        recursive = FALSE,
        ignore.case = FALSE,
        include.dirs = FALSE)
    remove_summary_suffix <- function(x) { return(gsub("__summary", "", x)) }
    input_filenames <- lapply(summary_files, remove_summary_suffix)
    invisible(lapply(input_filenames, loadData))
}

### -------- Saving Plots --------
outpath <- "../"

default_plot_width=8
default_plot_height=6
default_plot_units="cm"
save_plot <- function(plot, name, path=outpath, width=default_plot_width,
    height=default_plot_height, units=default_plot_units) {
    filename <- paste(path, name, ".pdf", sep="")
    # print(paste("Plot saved at", filename))
    ggsave(plot, file=filename, width=width, height=height, units=units)
}

mkdir <- function(path) {
    if (!dir.exists(path)) { dir.create(path) }
}

### -------- Operations on input data --------
determine_threshold_epoch <- function(epochData, loss_threshold) {
    # add loss_threshold column to dataframe
    data <- merge(epochData, loss_threshold, by = "dataset")

    # determine first epoch under threshold (i.e. with loss <= loss_threshold) for each measurement
    data <- melt(data, id = c(data_id, "epoch", "loss_threshold"))
    data <- subset(data, variable == "validationLoss") # remove unused values
    data <- data[, names(data) != "variable"] # remove unnecessary column
    last_epochs <- subset(data, epoch == epochCount)
    data <- subset(data, value <= loss_threshold) # remove all epochs over the threshold
    data <- rbind(data, last_epochs)
    data <- data[, !(names(data) %in% c("value", "loss_threshold"))] # remove unnecessary column
    # determine min epoch (under the threshold, for each measurement)
    data <- melt(data, id = data_id)
    threshold_epoch <- dcast(data, ... ~ variable, fun.aggregate = min)
    names(threshold_epoch)[names(threshold_epoch) == 'epoch'] <- 'threshold_epoch'
    return(threshold_epoch)
}

filter_epochs_by_loss_threshold <- function(epochData, threshold_epoch) {
    # add threshold_epoch column to dataframe
    data <- merge(epochData, threshold_epoch, by = data_id)
    filtered_data <- subset(data, epoch <= threshold_epoch) # remove all epochs after the threshold_epoch
    return(filtered_data)
}

aggregate_epoch_times <- function(epochData, aggregation) { # pass e.g. sum or mean for aggregation parameter
    molten_epochData <- melt(epochData, id = data_id)
    molten_epochData <- subset(molten_epochData, variable %in% c("totalTime", "computationTime", "communicationTime"))
    aggregatedData <- dcast(molten_epochData, ... ~ variable, fun.aggregate = aggregation)
    return(aggregatedData)
}

prepare_speedup_data <- function(data, threshold_epochs) {
    data <- filter_epochs_by_loss_threshold(data, threshold_epochs)

    # aggregate data
    moltenEpochData <- melt(data, id = data_id)
    moltenEpochData <- subset(moltenEpochData, variable == "totalTime")
    moltenEpochData$value <- as.numeric(moltenEpochData$value)

    totalTrainingTime <- dcast(moltenEpochData, ... ~ variable, fun.aggregate = sum)
    names(totalTrainingTime)[names(totalTrainingTime) == 'totalTime'] <- 'total_training_time'
    avgEpochTime <- dcast(moltenEpochData, ... ~ variable, fun.aggregate = mean)
    names(avgEpochTime)[names(avgEpochTime) == 'totalTime'] <- 'mean_epoch_time'
    data <- merge(totalTrainingTime, avgEpochTime, by = data_id)

    # extract and filter parallel data
    min_batch_size <- min(data$mini_batch_size) * max(data$nodeCount)
    data_batch_size <- subset(data, batch_size >= min_batch_size & training == "AllReduce")

    # extract sequential data and merge with parallel data
    sequentialData <- subset(data, training == "Sequential")
    names(sequentialData)[names(sequentialData) == 'total_training_time'] <- 'seq_total_training_time'
    names(sequentialData)[names(sequentialData) == 'mean_epoch_time'] <- 'seq_mean_epoch_time'
    ignore <- c("training", "nodeCount", "date", "mini_batch_size", "epochCount")
    seqData_batch_size <- sequentialData[, !(names(sequentialData) %in% ignore)]
    merge_by_batch_size <- data_id[-which(data_id %in% ignore)]

    # merge parallel with sequential data and compute speedup and efficiency
    data_batch_size <- merge(data_batch_size, seqData_batch_size, by = merge_by_batch_size)
    max_p <- max(data$nodeCount)
    data_batch_size["total_speedup"] <- data_batch_size$seq_total_training_time / data_batch_size$total_training_time
    data_batch_size["total_efficiency"] <- data_batch_size$total_speedup / data_batch_size$nodeCount * max_p
    data_batch_size["mean_speedup"] <- data_batch_size$seq_mean_epoch_time / data_batch_size$mean_epoch_time
    data_batch_size["mean_efficiency"] <- data_batch_size$mean_speedup / data_batch_size$nodeCount * max_p
    data_batch_size <- melt(data_batch_size, id = data_id)

    return(data_batch_size)
}

### -------- Reusable Plot configurations --------
col_vec = c("red2","dodgerblue2","darkorange","chartreuse3","black", "magenta4", "turquoise", "grey45")
col_gradient = c("darkorange", "red2", "magenta4", "dodgerblue2", "turquoise", "chartreuse3", "grey45", "black")

plot_theme <- function(...) {
    list(
        theme_classic(...),
        theme(plot.title = element_text(size=15, margin=margin(t=10, b=10)), legend.key=element_blank()),
        guides(colour = guide_legend(ncol = 1))
    )
}

line_plot <- function(...) {
    list(
        expand_limits(y = 0),
        geom_line(...),
        geom_point(...)
    )
}

barchart <- function(aggr="identity", ...) {
    list(
        stat_summary(geom="bar", position="stack", width=0.9, fun.y=aggr, ...),
        expand_limits(y=0)
    )
}

percentage_scale <- scale_y_continuous(labels = scales::percent)
model_device_facet <- function(...) { return(facet_wrap(model~device, nrow = 2, ncol = 2, ...)) }
model_dataset_facet <- function(...) { return(facet_wrap(model~dataset, nrow = 2, ncol = 2, ...)) }

generalPlotConfigs <- function(plot) {
    plot <- plot + plot_theme() +
        scale_colour_manual(values=col_vec)
    return(plot)
}

genericLinePlot <- function(plot) {
    plot <- generalPlotConfigs(plot) + line_plot()
    return(plot)
}

pow2s <- c(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
wide_legend <- guide_legend(keywidth = 3.3, keyheight = 1, ncol = 1)
speedup_plot <- function(maxThreads) {
    shape_values = c("total_speedup" = 19, "mean_speedup" = 19, "total_efficiency" = NULL, "mean_efficiency" = NULL)
    linetype_values = c("total_speedup" = "solid", "mean_speedup" = "solid", "total_efficiency" = "dashed",
        "mean_efficiency" = "dashed")
    efficiency_alpha = 0.6
    alpha_values = c("total_speedup" = 1, "mean_speedup" = 1, "total_efficiency" = efficiency_alpha,
        "mean_efficiency" = efficiency_alpha)
    labels = c("total_speedup" = "Speedup", "mean_speedup" = "Speedup", "total_efficiency" = "Efficiency",
        "mean_efficiency" = "Efficiency")
    list(
        geom_line(),
        geom_point(),
        scale_shape_manual("", values = shape_values, labels = labels, guide=wide_legend),
        scale_linetype_manual("", values = linetype_values, labels = labels, guide=wide_legend),
        scale_alpha_manual("", values = alpha_values, labels = labels, guide=wide_legend),
        geom_abline(intercept = 0, slope = 1, linetype = "dotted", colour="grey"),
        coord_fixed(ratio=1, ylim=c(0.5,maxThreads), xlim=c(1,maxThreads)),
        scale_y_continuous(breaks=pow2s, sec.axis = sec_axis(~./maxThreads, name = "Efficiency")),
        scale_x_continuous(breaks=pow2s)
    )
}

