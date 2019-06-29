library(ggplot2)
library(reshape2)
library(RColorBrewer)
Sys.setenv(LANG = "en")

### -------- Constants --------

inpath <- "../../implementation/outputs/"
outpath <- "../"

col_vec = c("red2","darkorange","dodgerblue2","black","chartreuse3", "magenta4", "turquoise", "grey45")
col_gradient = c("darkorange", "red2", "deeppink", "magenta4", "dodgerblue2", "turquoise", "chartreuse3", "grey45", "black")

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
                 dataset=character(),
                 dataLoader=character(),
                 model=character(),
                 training=character(),
                 learning_rate=double(),
                 momentum=double(),
                 date=character(),
                 total_training_time=double())


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
                 date=character())

### -------- Functions --------

loadData <- function(name) {
    summary  <- read.table(paste(inpath, name, "__summary", sep=""), header=TRUE)
    epochs <- read.table(paste(inpath, name, "__epochs", sep=""), header=TRUE)

    summary['mini_batch_size'] <- summary$batch_size / summary$nodeCount
    epochs["summedTotalTime"] <- cumsum(epochs$totalTime) # Reduce("+", epochs["totalTime"], accumulate = TRUE)

    epochs["batch_size"]          <- rep(summary$batch_size[[1]],nrow(epochs))
    epochs["mini_batch_size"]     <- rep(summary$mini_batch_size[[1]],nrow(epochs))
    epochs["nodeCount"]           <- rep(summary$nodeCount[[1]],nrow(epochs))
    epochs["dataset"]             <- rep(summary$dataset[[1]],nrow(epochs))
    epochs["dataLoader"]          <- rep(summary$dataLoader[[1]],nrow(epochs))
    epochs["model"]               <- rep(summary$model[[1]],nrow(epochs))
    epochs["training"]            <- rep(summary$training[[1]],nrow(epochs))
    epochs["learning_rate"]       <- rep(summary$learning_rate[[1]],nrow(epochs))
    epochs["momentum"]            <- rep(summary$momentum[[1]],nrow(epochs))
    epochs["date"]                <- rep(summary$date[[1]],nrow(epochs))
    epochs["total_training_time"] <- rep(summary$totalTime[[1]],nrow(epochs))

    if (summary$training[[1]] == "Sequential") {
        epochs$computationTime <- epochs$totalTime
    }

    epochData <<- rbind(epochData, epochs)
    summaryData <<- rbind(summaryData, summary)
}

savePlot <- function(plot, name, path=outpath, width=8, height=6) {
    ggsave(plot, file=paste(path, name, ".pdf", sep=""), width=width, height=height)
}

compute_seq_time <- function(epochData, summaryData) {
    seq_epochData <- subset(epochData, training == "Sequential")
    seq_summaryData <- subset(summaryData, training == "Sequential")

    seq_epoch_time <- aggregate(totalTime ~ batch_size, data = seq_epochData, mean)
    names(seq_epoch_time)[names(seq_epoch_time) == 'totalTime'] <- 'mean_seq_epoch_time'

    seq_training_time <- subset(seq_summaryData, training == "Sequential")
    seq_training_time <- seq_training_time[c("batch_size", "totalTime")]
    names(seq_training_time)[names(seq_training_time) == 'totalTime'] <- 'seq_training_time'

    seq_time <- merge(seq_epoch_time, seq_training_time, by = "batch_size")
    seq_time["mini_batch_size"] <- seq_time$batch_size
    return(seq_time)
}

generalPlotConfigs <- function(plot) {
    plot <- plot +
        theme_classic() +
        scale_colour_manual(values=col_vec) +
        theme(plot.title = element_text(size=15, margin=margin(t=10, b=10)), legend.key=element_blank()) +
        guides(colour = guide_legend(nrow = 2))
    return(plot)
}

genericLinePlot <- function(plot) {
    plot <- generalPlotConfigs(plot) +
        expand_limits(y = 0) +
        geom_line() +
        geom_point()
    return(plot)
}

percentagePlot <- function(plot) {
    return(plot + scale_y_continuous(labels = scales::percent))
}

genericSpeedupPlot <- function(plot, maxThreads) {
    plot <- plot +
        geom_abline(intercept = 0, slope = 1) +
        coord_fixed(ylim=c(0,maxThreads), xlim=c(0,maxThreads)) +
        scale_y_continuous(breaks=c(1,2,4,8,16,32,64)) +
        scale_x_continuous(breaks=c(1,2,4,8,16,32,64))
    return(plot)
}

