# Extract tabular analysis copies from untouched spData RDA originals.
# Run from the repository root using Rscript --vanilla.
# Only base R is required. No source code from the data provider is executed.

options(digits = 17)
output <- "Data/External/Processed"
dir.create(output, recursive = TRUE, showWarnings = FALSE)

write_precise_csv <- function(frame, destination) {
  # Seventeen significant digits preserve binary64 numerical roundtrips.
  exported <- frame
  for (column in seq_along(exported)) {
    if (is.numeric(exported[[column]])) {
      exported[[column]] <- sprintf("%.17g", exported[[column]])
    }
  }
  write.table(exported, destination, sep = ",", row.names = FALSE,
              col.names = TRUE, quote = TRUE, na = "NA", fileEncoding = "UTF-8",
              eol = "\n")
}

for (dataset in c("columbus", "baltimore", "boston")) {
  objects <- new.env(parent = baseenv())
  load(file.path("Data/External/Raw/spData/data", paste0(dataset, ".rda")),
       envir = objects)
  cat("DATASET", dataset, "OBJECTS", ls(objects), "\n")
  if (dataset == "boston") {
    frame <- objects$boston.c
    coordinates <- objects$boston.utm
    stopifnot(nrow(frame) == nrow(coordinates))
    if (!is.null(rownames(coordinates))) {
      stopifnot(identical(rownames(frame), rownames(coordinates)))
    }
    cat("COORDINATES boston.utm; dimensions", dim(coordinates),
        "; coordinate names", colnames(coordinates), "\n")
    frame$X <- coordinates[, 1]
    frame$Y <- coordinates[, 2]
  } else {
    frame <- get(dataset, envir = objects)
  }
  original_names <- names(frame)
  names(frame) <- make.unique(original_names, sep = "_duplicate_")
  mapping <- data.frame(position = seq_along(original_names),
                        source_name = original_names, output_name = names(frame))
  write_precise_csv(mapping, file.path(output, paste0(dataset, "_columns.csv")))
  frame <- data.frame(observation_id = rownames(frame), frame, check.names = FALSE)
  stopifnot(!anyDuplicated(frame$observation_id))
  destination <- file.path(output, paste0(dataset, ".csv"))
  write_precise_csv(frame, destination)
  roundtrip <- read.csv(destination, check.names = FALSE)
  for (column in seq_along(frame)) {
    if (is.numeric(frame[[column]])) {
      stopifnot(identical(as.double(frame[[column]]),
                          as.double(roundtrip[[column]])))
    }
  }
  cat("SOURCE_NUMERICAL_ROUNDTRIP_OK", dataset, "\n")
  cat("ROWS", nrow(frame), "COLUMNS", names(frame), "\n")
}
capture.output(sessionInfo(), file = file.path(output, "r_session.txt"))
