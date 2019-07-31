library(ggplot2)
library(reshape2)
library(RColorBrewer)
Sys.setenv(LANG = "en")
source("utils.R")

### -------- Plot functions --------
speedup_plot_helper <- function(data, maxThreads, title, color, color_label, facet=NULL, gradient=TRUE) {
    data["color"] <- data[color]

    cols <- col_vec
    if (gradient) { cols <- col_gradient }

    plot <- ggplot(data, aes(x=nodeCount, y=value, color=as.factor(color), shape=variable, linetype=variable, alpha=variable)) +
            speedup_plot(maxThreads) +
            facet +
            plot_theme() +
            scale_colour_manual(values=cols) +
            labs(x="p", y="Speedup", color=color_label)
    if (!is.null(title)) { plot <- plot + labs(title=title) }
    return(plot)
}


speedupEfficiencyPlots <- function(data, threshold_epochs) {
    data <- prepare_speedup_data(data, threshold_epochs)

    ### Plot configs
    maxThreads <- max(epochData$nodeCount)
    plotHelper <- function(data, title) {
        plot <- ggplot(data, aes(x=nodeCount, y=value, color=as.factor(batch_size), shape=variable, linetype=variable)) +
                speedup_plot(maxThreads) +
                model_device_facet() +
                plot_theme() +
                scale_colour_manual(values=col_gradient) +
                labs(title=title, x="p", y="Speedup", color="Batch Size")
        return(plot)
    }

    ### Actual plots
    meanPlot <- plotHelper(subset(data, variable %in% c("mean_speedup", "mean_efficiency")),
        title="Mean Epoch Speedup with Equal Batch Size")

    totalPlot <- plotHelper(subset(data, variable %in% c("total_speedup", "total_efficiency")),
        title="Total Speedup with Equal Batch Size")

    return(list(meanPlot, totalPlot))
}

aggregate_comp_comm_time <- function(data, id_vars, aggregation) {
    # aggregate and filter data
    data <- melt(data, id.vars = id_vars)
    data <- subset(data, variable %in% c("computationTime", "communicationTime"))
    data$value <- as.numeric(data$value)
    aggr_data <- dcast(data, ... ~ variable, fun.aggregate = aggregation)
    aggr_data <- melt(aggr_data, id.vars = id_vars)

    # reorder factor levels
    levels <- c("communicationTime", "computationTime")
    aggr_data <- within(aggr_data, variable <- factor(variable, levels=levels))

    return(aggr_data)
}

comp_comm_time_plot <- function(data, title, group, group_label, facet=model_device_facet(scales="free"), shortLabels=FALSE) {
    long_labels <- c("communicationTime" = "Communication", "computationTime" = "Computation")
    short_labels <- c("communicationTime" = "Comm.", "computationTime" = "Comp.")
    if(shortLabels) { labels <- short_labels }
    else { labels <- long_labels }
    data["x"] <- data[group]
    if (!is.factor(data$x)) { data$x <- as.factor(data$x) }
    plot <- ggplot(data, aes(x=x, y=value, fill=as.factor(variable))) +
            barchart() +
            facet +
            plot_theme() +
            scale_fill_manual(values=col_vec, labels=labels) +
            labs(x=group_label, y="Time [s]", fill="")
    if (!is.null(title)) { plot <- plot + labs(title=title) }
    return(plot)
}

title_comm_mean <- "Training Time per Epoch"
title_comm_total <- "Total Training Time"

comm_by_training_helper <- function(data, title_extension, file_extension) {
    id_vars <- c(data_id, "training_node_count")
    labels <- c("Seq", "p 1", "p 2", "p 4", "p 8", "p 16")

    meanData <- aggregate_comp_comm_time(data, id_vars, mean)
    meanTitle <- paste(title_comm_mean, title_extension)
    meanPlot <- comp_comm_time_plot(meanData, meanTitle, group="training_node_count", group_label="Training") +
                    scale_x_discrete(labels=labels)
    save_plot(meanPlot, paste("CommunicationMean", file_extension, sep="-"))

    totalData <- aggregate_comp_comm_time(data, id_vars, sum)
    totalTitle <- paste(title_comm_total, title_extension)
    totalPlot <- comp_comm_time_plot(totalData, totalTitle, group="training_node_count", group_label="Training") +
                    scale_x_discrete(labels=labels)
    save_plot(totalPlot, paste("CommunicationTotal", file_extension, sep="-"))
}

comm_by_var_helper <- function(data, title_extension, path, file_extension, variable, var_label) {
    id_vars <- data_id[!data_id %in% c("batch_size", "mini_batch_size")]
    id_vars <- c(id_vars, variable, "training_node_count")

    meanData <- aggregate_comp_comm_time(data, id_vars, mean)
    meanTitle <- paste(title_comm_mean, title_extension)
    meanPlot <- comp_comm_time_plot(meanData, meanTitle, group=variable, group_label=var_label)
    # meanPlot <- meanPlot + scale_x_continuous(breaks=pow2s)
    save_plot(meanPlot, paste("CommunicationMean", file_extension, sep="-"), path=path)

    totalData <- aggregate_comp_comm_time(data, id_vars, sum)
    totalTitle <- paste(title_comm_total, title_extension)
    totalPlot <- comp_comm_time_plot(totalData, totalTitle, group=variable, group_label=var_label)
    # totalPlot <- totalPlot + scale_x_continuous(breaks=pow2s)
    save_plot(totalPlot, paste("CommunicationTotal", file_extension, sep="-"), path=path)
}

loss_plot <- function(data, title, color, color_label, shape=NULL, shape_label="", x="epoch", x_label="Epochs", gradient=TRUE) {
    data["color"] <- data[color]
    data["x"] <- data[x]

    data["plot_point"] <- ifelse(data$dataset == "MNIST", TRUE, data$epoch %% 5 == 0)

    cols <- col_vec
    if (gradient) { cols <- col_gradient }

    if (!is.null(shape)) {
        data["shape"] <- data[shape]
        plot <- ggplot(data, aes(x=x, y=validationLoss, color=as.factor(color), shape=as.factor(shape)))
    } else {
        plot <- ggplot(data, aes(x=x, y=validationLoss, color=as.factor(color), shape=NULL))
    }
    plot <- plot +
        plot_theme() +
        geom_line() +
        geom_point(data=subset(data, plot_point)) +
        expand_limits(y = 0) +
        scale_colour_manual(values=cols) +
        labs(x=x_label, y="Validation Loss", color=color_label, shape=shape_label)
    if (!is.null(title)) { plot <- plot + labs(title=title) }
    return(plot)
}

title_loss <- "Validation Loss per Epoch"

loss_by_training_helper <- function(data, title_extension, file_extension, facet) {
    plot <- loss_plot(data, paste(title_loss, title_extension),
        color="training_node_count", color_label="Training") + facet
    save_plot(plot, paste("ValidationLoss", file_extension, sep="-"))
}

loss_by_var_helper <- function(data, title_extension, path, file_extension, variable, var_label, facet) {
    plot <- loss_plot(data, paste(title_loss, title_extension),
        color=variable, color_label=var_label) + facet
    save_plot(plot, paste("ValidationLoss", file_extension, sep="-"), path=path)
}