library(ggplot2)
library(reshape2)
library(RColorBrewer)
Sys.setenv(LANG = "en")
source("utils.R")


linePlot <- function(data, x_axis, y_axis, x_label, y_label, group_label, title) {
    plot <- ggplot(data, aes_string(x=x_axis, y=y_axis, colour='group', group='group')) +
        labs(title=title, x=x_label, y=y_label, color=group_label, shape=group_label)
    plot <- genericLinePlot(plot)

    return(plot)
}

commTimePlot <- function(data, x_label, title, stat) {
    id_vars <- c("batch_size", "mini_batch_size", "nodeCount", "dataset", "dataLoader", "model",
        "training", "learning_rate", "momentum", "date", "total_training_time", "group")
    data <- melt(data, id.vars = id_vars)
    data <- subset(data, variable %in% c("computationTime", "communicationTime"))
    data <- within(data, variable <- factor(variable, levels=c("communicationTime", "computationTime")))
    labels <- c("communicationTime" = "Communication", "computationTime" = "Computation")

    plot <- ggplot(data) +
        stat_summary(aes(x = group, y = value, fill=variable), fun.y = stat,
            geom = "bar", position = "stack", width = 0.9) +
        scale_fill_manual(name="", values=col_vec, labels = labels) +
        expand_limits(y = 0) +
        labs(title=title, x=x_label, y="Time [s]")
    plot <- generalPlotConfigs(plot)

    return(plot)
}

totalTimePlot <- function(data, title) {
    data["commPlusComp"] <- data$computationTime+data$communicationTime
    id_vars <- c("batch_size", "mini_batch_size", "nodeCount", "dataset", "dataLoader", "model",
        "training", "learning_rate", "momentum", "date", "total_training_time", "group", "epoch")
    data <- melt(data, id.vars = id_vars)
    data <- subset(data, variable %in% c("commPlusComp", "totalTime"))
    labels <- c("totalTime" = "Total Training Time", "commPlusComp" = "Computation + Communication Time")

    plot <- ggplot(data) +
        stat_summary(aes(x=epoch, y=value, color=group, shape=variable), geom = "point", fun.y = "identity") +
        stat_summary(aes(x=epoch, y=value, color=group, shape=variable), geom = "line", fun.y = "identity") +
        expand_limits(y = 0) +
        labs(title=title, x="Epoch", y="Time [s]", color=group_label, shape="Time")+
        theme_classic() +
        scale_colour_manual(values=col_vec, labels=labels) +
        theme(plot.title = element_text(size=15, margin=margin(t=10, b=10)), legend.key=element_blank()) +
        guides(colour = guide_legend(nrow = 2))

}

### -------- Configuration --------

loadData("results__2019-06-27--14-36-07") # Sequential, BS  32, p=1
loadData("results__2019-06-27--12-34-23") # Sequential, BS  64, p=1
loadData("results__2019-06-27--14-26-09") # Sequential, BS 128, p=1

loadData("results__2019-06-28--10-31-05") # AllReduce,  BS  32, p=1
loadData("results__2019-06-28--10-36-01") # AllReduce,  BS  64, p=1
loadData("results__2019-06-28--10-40-32") # AllReduce,  BS 128, p=1

loadData("results__2019-06-28--10-45-19") # AllReduce,  BS  64, p=2
loadData("results__2019-06-28--10-49-18") # AllReduce,  BS 128, p=2

data <- epochData

group_label <- "Training and Process Count"
data["group"] <- paste(data$training, data$nodeCount)

### -------- Actual plots --------

plotLinePlots <- function(data, title_extension, file_name) {
    title = paste("Validation Error per Epoch", title_extension, sep="")
    errorByEpoch <- linePlot(data, "epoch", "validationError", "Epoch", "Validation Error", group_label, title)
    errorByEpoch <- percentagePlot(errorByEpoch)
    savePlot(errorByEpoch, paste("errorByEpoch/", file_name, sep=""))

    title = paste("Validation Error by Training Time", title_extension, sep="")
    errorByTime <- linePlot(data, "summedTotalTime", "validationError", "Training Time [s]", "Validation Error", group_label, title)
    errorByTime <- percentagePlot(errorByTime)
    savePlot(errorByTime, paste("errorByTime/", file_name, sep=""))

    title = paste("Validation Loss per Epoch", title_extension, sep="")
    validationLossByEpoch <- linePlot(data, "epoch", "validationLoss", "Epoch", "Validation Loss", group_label, title)
    savePlot(validationLossByEpoch, paste("validationLossByEpoch/", file_name, sep=""))

    title = paste("Validation Loss by Training Time", title_extension, sep="")
    validationLossByTime <- linePlot(data, "summedTotalTime", "validationLoss", "Training Time [s]", "Validation Loss", group_label, title)
    savePlot(validationLossByTime, paste("validationLossByTime/", file_name, sep=""))

    data <- subset(data, epoch!=0)
    title = paste("Training Loss per Epoch", title_extension, sep="")
    trainingLossByEpoch <- linePlot(data, "epoch", "trainingLoss", "Epoch", "Training Loss", group_label, title)
    savePlot(trainingLossByEpoch, paste("trainingLossByEpoch/", file_name, sep=""))

    title = paste("Training Loss by Training Time", title_extension, sep="")
    trainingLossByTime <- linePlot(data, "summedTotalTime", "trainingLoss", "Training Time [s]", "Training Loss", group_label, title)
    savePlot(trainingLossByTime, paste("trainingLossByTime/", file_name, sep=""))


    title = paste("Training Time per Epoch", title_extension, sep="")
    timePerEpoch <- linePlot(data, "epoch", "totalTime", "Epoch", "Training Time [s]", group_label, title)
    savePlot(timePerEpoch, paste("timePerEpoch/", file_name, sep=""))

    title = paste("Share of Communication Time per Epoch", title_extension, sep="")
    trainingLossByEpoch <- commTimePlot(data, group_label, title, "mean")
    savePlot(trainingLossByEpoch, paste("commTimeEpoch/", file_name, sep=""))

    title = paste("Total Share of Communication Time", title_extension, sep="")
    trainingLossByTime <- commTimePlot(data, group_label, title, "sum")
    savePlot(trainingLossByTime, paste("commTimeTotal/", file_name, sep=""))


    title = paste("Computation Time vs. Total Time", title_extension, sep="")
    compVsTotalTime <- totalTimePlot(data, title)
    savePlot(compVsTotalTime, paste("compVsTotalTime/", file_name, sep=""))
}

plotWithBatchSize <- function(batch_size) {
    local_batch_size <- batch_size
    data <- subset(data, batch_size == local_batch_size)
    title_extension = paste(" with Batch Size ", batch_size, sep="")
    file_name = paste("batch_size-", batch_size, sep="")
    return(plotLinePlots(data, title_extension, file_name))
}

plotWithMiniBatchSize <- function(mini_batch_size) {
    local_mini_batch_size <- mini_batch_size
    data <- subset(data, mini_batch_size == local_mini_batch_size)
    title_extension = paste(" with Mini-Batch Size ", mini_batch_size, sep="")
    file_name = paste("mini_batch_size-", mini_batch_size, sep="")
    return(plotLinePlots(data, title_extension, file_name))
}


plotWithMiniBatchSize(32)
plotWithMiniBatchSize(64)

plotWithBatchSize(64)
plotWithBatchSize(128)