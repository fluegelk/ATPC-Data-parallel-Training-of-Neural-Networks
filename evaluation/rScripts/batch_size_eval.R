library(ggplot2)
library(reshape2)
library(RColorBrewer)
Sys.setenv(LANG = "en")

### -------- Constants --------

inpath <- "../../implementation/outputs/batch_size_eval/"
outpath <- "../batch_size_eval/"

col_vec = c("red2","darkorange","dodgerblue2","black","chartreuse3", "magenta4", "turquoise", "grey45")
col_gradient = c("darkorange", "red2", "deeppink", "magenta4", "dodgerblue2", "turquoise", "chartreuse3", "grey45", "black")
col_bs <- c("512" = "darkorange", "256" = "red2", "128" = "deeppink", "64" = "magenta4", "32" = "dodgerblue2", "16" = "turquoise", "8" = "chartreuse3", "4" = "grey45")

epochData <- data.frame(SGD=character(),
                 model=character(),
                 dataset=character(),
                 batchSize=integer(),
                 nodeCount=integer(),
                 epoch=integer(),
                 totalTime=double(),
                 computationTime=double(),
                 communicationTime=double(),
                 validationError=double(),
                 validationLoss=double(),
                 trainingLoss=double(),
                 summedTotalTime=double())
summaryData <- data.frame(SGD=character(),
                 model=character(),
                 dataset=character(),
                 batchSize=integer(),
                 nodeCount=integer(),
                 epochCount=integer(),
                 totalTime=double(),
                 validationError=double(),
                 validationLoss=double())

### -------- Functions --------

loadData <- function(name, SGD, model, dataset) {
    summary  <- read.table(paste(inpath, name, "__summary", sep=""), header=TRUE)
    summary$batchSize <- as.integer(summary$batchSize)
    summary$nodeCount <- as.integer(summary$nodeCount)
    summary$epochCount <- as.integer(summary$epochCount)

    epochs <- read.table(paste(inpath, name, "__epochs", sep=""), header=TRUE)
    epochs["summedTotalTime"] <- cumsum(epochs$totalTime) # Reduce("+", epochs["totalTime"], accumulate = TRUE)
    epochs["batchSize"] <- c(summary$batchSize[[1]])
    epochs["nodeCount"] <- c(summary$nodeCount[[1]])

    summary["SGD"] <- c(SGD)
    epochs["SGD"] <- c(SGD)
    summary["model"] <- c(model)
    epochs["model"] <- c(model)
    summary["dataset"] <- c(dataset)
    epochs["dataset"] <- c(dataset)

    epochData <<- rbind(epochData, epochs)
    summaryData <<- rbind(summaryData, summary)
}

linePlot <- function(data, x_axis, y_axis, group, x_label, y_label, group_label, title) {
    plot <- ggplot(data, aes_string(x=x_axis, y=y_axis, colour=group, group=group)) +
        theme_classic() +
        facet_wrap(dataset~model,scales="free", nrow=2) +
        expand_limits(y = 0) +
        geom_line() +
        geom_point() +
        scale_colour_manual(values=col_bs) +
        labs(title=title, x=x_label, y=y_label, color=group_label, shape=group_label) +
        theme(plot.title = element_text(size=15, margin=margin(t=10, b=10)), legend.key=element_blank()) +
        guides(colour = guide_legend(ncol = 1))

    return(plot)
}
# optional: add to plot for percentage y axis
# + scale_y_continuous(labels = scales::percent)

savePlot <- function(plot, name) {
    ggsave(plot, file=paste(outpath, name, ".pdf", sep=""), width=12, height=6)
}

### -------- Configuration --------
models=c("AlexNet", "LeNet5", "LeNet5Updated", "PyTorchTutorialNet")
batchSizes=c(4,8,16,32,64,128,256,512)
datasets=c("MNIST", "CIFAR10")

for (model in models) {
    for (batchSize in batchSizes) {
        for (dataset in datasets) {
            filename = paste("results__", model, "__", dataset, "__batchsize-", batchSize, sep="")
            loadData(filename, "Sequential", model, dataset)
        }
    }
}
data <- subset(epochData, epoch <= 10)
data['batchSizeFactor'] <- as.factor(data$batchSize)

group <- "batchSizeFactor"
group_label <- "Batch Size"

### -------- Actual plots --------

title = "Validation Error per Epoch"
errorByEpoch <- linePlot(data, "epoch", "validationError", group, "Epoch", "Validation Error", group_label, title)
errorByEpoch <- errorByEpoch + scale_y_continuous(labels = scales::percent)
errorByEpoch <- errorByEpoch + scale_x_continuous(breaks=c(0,2,4,6,8,10))
savePlot(errorByEpoch, "errorByEpoch")

title = "Validation Loss per Epoch"
validationLossByEpoch <- linePlot(data, "epoch", "validationLoss", group, "Epoch", "Validation Loss", group_label, title)
validationLossByEpoch <- validationLossByEpoch + scale_x_continuous(breaks=c(0,2,4,6,8,10))
savePlot(validationLossByEpoch, "validationLossByEpoch")

data <- subset(data, epoch > 0)
title = "Training Time per Epoch"
timePerEpoch <- linePlot(data, "epoch", "totalTime", group, "Epoch", "Training Time [s]", group_label, title)
timePerEpoch <- timePerEpoch + scale_x_continuous(breaks=c(0,2,4,6,8,10))
savePlot(timePerEpoch, "timePerEpoch")

