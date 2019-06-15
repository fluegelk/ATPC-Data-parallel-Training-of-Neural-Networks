library(ggplot2)
library(reshape2)
library(RColorBrewer)
Sys.setenv(LANG = "en")

### -------- Constants --------

inpath <- "../../implementation/outputs/"
outpath <- "../"

col_vec = c("red2","darkorange","dodgerblue2","black","chartreuse3", "magenta4", "turquoise", "grey45")
col_gradient = c("darkorange", "red2", "deeppink", "magenta4", "dodgerblue2", "turquoise", "chartreuse3", "grey45", "black")

epochData <- data.frame(SGD=character(),
                 batchSize=double(),
                 nodeCount=double(),
                 epoch=double(),
                 totalTime=double(),
                 computationTime=double(),
                 communicationTime=double(),
                 validationError=double(),
                 validationLoss=double(),
                 trainingLoss=double(),
                 summedTotalTime=double())
summaryData <- data.frame(SGD=character(),
                 batchSize=double(),
                 nodeCount=double(),
                 epochCount=double(),
                 totalTime=double(),
                 validationError=double(),
                 validationLoss=double())

### -------- Functions --------

loadData <- function(name, SGD) {
    summary  <- read.table(paste(inpath, name, "__summary", sep=""), header=TRUE)
    epochs <- read.table(paste(inpath, name, "__epochs", sep=""), header=TRUE)

    epochs["summedTotalTime"] <- cumsum(epochs$totalTime) # Reduce("+", epochs["totalTime"], accumulate = TRUE)
    epochs["batchSize"] <- c(summary$batchSize[[1]])
    epochs["nodeCount"] <- c(summary$nodeCount[[1]])

    summary["SGD"] <- c(SGD)
    epochs["SGD"] <- c(SGD)

    epochData <<- rbind(epochData, epochs)
    summaryData <<- rbind(summaryData, summary)
}

linePlot <- function(data, x_axis, y_axis, group, x_label, y_label, group_label) {
    plot <- ggplot(data, aes_string(x=x_axis, y=y_axis, colour=group, group=group)) +
        theme_classic() +
        expand_limits(y = 0) +
        geom_line() +
        geom_point() +
        scale_colour_manual(values=col_vec) +
        labs(x=x_label, y=y_label, color=group_label, shape=group_label) +
        theme(plot.title = element_text(size=15, margin=margin(t=10, b=10)), legend.key=element_blank()) +
        guides(colour = guide_legend(nrow = 2))

    return(plot)
}
# optional: add to plot for percentage y axis
# + scale_y_continuous(labels = scales::percent)

savePlot <- function(plot, name) {
    ggsave(plot, file=paste(outpath, name, ".pdf", sep=""), width=8, height=6)
}

### -------- Configuration --------

loadData("results__2019-06-15--12-05-43", "Sequential")

group <- "SGD"
group_label <- "SGD"
data <- epochData

### -------- Actual plots --------

errorByEpoch <- linePlot(data, "epoch", "validationError", group, "Epoch", "Validation Error", group_label)
errorByEpoch <- errorByEpoch + scale_y_continuous(labels = scales::percent)
savePlot(errorByEpoch, "errorByEpoch")

errorByTime <- linePlot(data, "summedTotalTime", "validationError", group, "Training Time [s]", "Validation Error", group_label)
errorByTime <- errorByTime + scale_y_continuous(labels = scales::percent)
savePlot(errorByTime, "errorByTime")


validationLossByEpoch <- linePlot(data, "epoch", "validationLoss", group, "Epoch", "Validation Loss", group_label)
savePlot(validationLossByEpoch, "validationLossByEpoch")

validationLossByTime <- linePlot(data, "summedTotalTime", "validationLoss", group, "Training Time [s]", "Validation Loss", group_label)
savePlot(validationLossByTime, "validationLossByTime")


trainingLossByEpoch <- linePlot(data, "epoch", "trainingLoss", group, "Epoch", "Training Loss", group_label)
savePlot(trainingLossByEpoch, "trainingLossByEpoch")

trainingLossByTime <- linePlot(data, "summedTotalTime", "trainingLoss", group, "Training Time [s]", "Training Loss", group_label)
savePlot(trainingLossByTime, "trainingLossByTime")


timePerEpoch <- linePlot(data, "epoch", "totalTime", group, "Epoch", "Training Time [s]", group_label)
savePlot(timePerEpoch, "timePerEpoch")

