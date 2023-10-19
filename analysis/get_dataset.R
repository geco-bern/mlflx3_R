# sequence to sequence
# dataset generator
# multiple y outputs
# need to verify the format with the python code
#
# initialize = function(indices) {
#   data <- self$prepare_mushroom_data(mushroom_data[indices, ])
#   self$xcat <- data[[1]][[1]]
#   self$xnum <- data[[1]][[2]]
#   self$y <- data[[2]]
# },
#
# .getitem = function(i) {
#   xcat <- self$xcat[i, ]
#   xnum <- self$xnum[i, ]
#   y <- self$y[i, ]
#
#   list(x = list(xcat, xnum), y = y)
# },
#
#  all <- torch_cat(list(embedded, xnum$to(dtype = torch_float())), dim = 2)
#  output <- model(b$x[[1]]$to(device = device), b$x[[2]]$to(device = device))

gpp_dataset <- dataset(
    name = "dataset",
    initialize = function(
      x,
      train_center,
      sample_frac = 1
    ) {

      # select values
      train_mean <- train_center |>
        select(
          ends_with("_mean")
        ) |>
        unlist() |>
        as.vector()

      train_sd <- train_center |>
        select(
          ends_with("_sd")
        ) |>
        unlist() |>
        as.vector()

      # pass on the sitename to constrain the subset
      self$sitename <- x |> select("sitename") |> unlist()
      self$sites <- unique(self$sitename)

      # split out the numeric data for centering
      x <- x |>
        ungroup() |>
        select(
          where(is.numeric)
        )

      # convert to matrix
      x <- as.matrix(x)

      # normalize columns in input matrix
      # note that apply transposes the data
      # cols become rows (documented behaviour but uggh)
      x <- t(apply(x, 1, function(x){(x - train_mean) / train_sd}))

      # convert to tensor
      self$x <- torch_tensor(x)

      # one hot encoding static variables
      # library(caret)
      #
      # #define one-hot encoding function
      # dummy <- dummyVars(" ~ .", data=df)
      #
      # #perform one-hot encoding on data frame
      # final_df <- data.frame(predict(dummy, newdata=df))

      self$c <- torch_tensor()

      # sample along the rows (time)
      self$idx <- sort(sample.int(
        n = length(self$sites),
        size = length(self$sites) * sample_frac
      ))
    },

    .getitem = function(i) {

      # find locations corresponding
      # to site locations only return a
      # site at a time (index value)
      l <- which(self$sitename == self$sites[self$idx[i]])

      # here the true training data is
      # generated, subsetting your target
      # y (defined by one position end
      # for a given )
      list(
        x = self$x[l,2:ncol(self$x)],
        y = self$x[l,1]
      )
    },

    # return length of the index
    # for internal bookkeeping
    .length = function() {
      length(self$idx)
    }
  )
