library(ggplot2)
library(reshape2)
library(RColorBrewer)
Sys.setenv(LANG = "en")
source("utils.R")

### -------- Configuration --------

loadData("results__2019-06-27--14-36-07") # Sequential, BS  32, p=1
loadData("results__2019-06-27--12-34-23") # Sequential, BS  64, p=1
loadData("results__2019-06-27--14-26-09") # Sequential, BS 128, p=1

loadData("results__2019-06-28--10-31-05") # AllReduce,  BS  32, p=1
loadData("results__2019-06-28--10-36-01") # AllReduce,  BS  64, p=1
loadData("results__2019-06-28--10-40-32") # AllReduce,  BS 128, p=1

loadData("results__2019-06-28--10-45-19") # AllReduce,  BS  64, p=2
loadData("results__2019-06-28--10-49-18") # AllReduce,  BS 128, p=2

seq_time <- compute_seq_time(epochData, summaryData)

par_data <- subset(epochData, training != "Sequential")
par_data['batch_size_factor'] <- as.factor(par_data$batch_size)
par_data['mini_batch_size_factor'] <- as.factor(par_data$mini_batch_size)

min_batch_size <- min(par_data$mini_batch_size) * max(par_data$nodeCount)
max_mini_batch_size <- max(par_data$batch_size) / max(par_data$nodeCount)

## Functions
epochSpeedupBatchSize <- function(data, mean_seq, title) {
    data <- subset(data, batch_size >= min_batch_size)
    data <- merge(data, mean_seq, by = "batch_size")
    plot <- ggplot(data, aes(x=nodeCount, y=mean_seq_epoch_time/mean(totalTime),
        colour=batch_size_factor, group=batch_size_factor)) +
        labs(title=title, x="p", y="Epoch Speedup", color="Batch Size", shape="Batch Size")
    plot <- genericLinePlot(plot)
    plot <- genericSpeedupPlot(plot, max(data$nodeCount))
    return(plot)
}

epochSpeedupMiniBatchSize <- function(data, mean_seq, title) {
    data <- subset(data, mini_batch_size <= max_mini_batch_size)
    data <- merge(data, mean_seq, by = "mini_batch_size")
    plot <- ggplot(data, aes(x=nodeCount, y=mean_seq_epoch_time/mean(totalTime),
        colour=mini_batch_size_factor, group=mini_batch_size_factor)) +
        labs(title=title, x="p", y="Epoch Speedup", color="Mini-Batch Size", shape="Mini-Batch Size")
    plot <- genericLinePlot(plot)
    plot <- genericSpeedupPlot(plot, max(data$nodeCount))
    return(plot)
}


totalSpeedupBatchSize <- function(data, mean_seq, title) {
    data <- subset(data, batch_size >= min_batch_size)
    data <- merge(data, mean_seq, by = "batch_size")
    plot <- ggplot(data, aes(x=nodeCount, y=seq_training_time/total_training_time,
        colour=batch_size_factor, group=batch_size_factor)) +
        labs(title=title, x="p", y="Total Speedup", color="Batch Size", shape="Batch Size")
    plot <- genericLinePlot(plot)
    plot <- genericSpeedupPlot(plot, max(data$nodeCount))
    return(plot)
}

totalSpeedupMiniBatchSize <- function(data, mean_seq, title) {
    data <- subset(data, mini_batch_size <= max_mini_batch_size)
    data <- merge(data, mean_seq, by = "mini_batch_size")
    plot <- ggplot(data, aes(x=nodeCount, y=seq_training_time/total_training_time,
        colour=mini_batch_size_factor, group=mini_batch_size_factor)) +
        labs(title=title, x="p", y="Total Speedup", color="Mini-Batch Size", shape="Mini-Batch Size")
    plot <- genericLinePlot(plot)
    plot <- genericSpeedupPlot(plot, max(data$nodeCount))
    return(plot)
}


### -------- Actual plots --------
title = "Mean Epoch Speedup with Equal Batch Size"
plot <- epochSpeedupBatchSize(par_data, seq_time, title)
savePlot(plot, "speedup/epochSpeedupBatchSize")

title = "Mean Epoch Speedup with Equal Mini-Batch Size"
plot <- epochSpeedupMiniBatchSize(par_data, seq_time, title)
savePlot(plot, "speedup/epochSpeedupMiniBatchSize")

title = "Total Speedup with Equal Batch Size"
plot <- totalSpeedupBatchSize(par_data, seq_time, title)
savePlot(plot, "speedup/totalSpeedupBatchSize")

title = "Total Speedup with Equal Mini-Batch Size"
plot <- totalSpeedupMiniBatchSize(par_data, seq_time, title)
savePlot(plot, "speedup/totalSpeedupMiniBatchSize")