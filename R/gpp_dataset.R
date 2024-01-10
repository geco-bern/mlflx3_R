#' Data format routine, queried by data loader
#'
#' @param x input dataset
#' @param train_center mean and sd to center the data
#'
#' @return torch tensor for model training
#' @export

gpp_dataset <- torch::dataset(
  name = "dataset",
  initialize = function(
    x,
    train_center
  ) {

    # select values from global mean and sd
    # to center the data properly - can't be
    # done internally as the test data should
    # be centered using the train data mean
    train_mean <- train_center |>
      select(
        ends_with("_mean")
      ) |>
      select(
        !starts_with("GPP")
      ) |>
      unlist() |>
      as.vector()

    train_sd <- train_center |>
      select(
        ends_with("_sd")
      ) |>
      select(
        !starts_with("GPP")
      ) |>
      unlist() |>
      as.vector()

    # pass on the sitename to constrain the subset
    self$sitename <- x |>
      dplyr::select("sitename") |>
      unlist()

    # forward unique index of sites to subset data
    self$sites <- unique(self$sitename)

    # target values
    y <- x |>
      ungroup() |>
      select(
        starts_with("GPP")
      ) |>
      unlist()

    self$y <- torch::torch_tensor(y)

    # split out the numeric data for centering
    x <- x |>
      ungroup() |>
      select(
        where(is.numeric)
      ) |>
      select(
        !starts_with("GPP")
      )

    # convert to matrix, data frame input
    # seems to mess with the tensor
    x <- as.matrix(x)

    # normalize columns in input matrix
    # note that apply transposes the data
    # cols become rows (documented behaviour but uggh)
    x <- t(apply(x, 1, function(x){(x - train_mean) / train_sd}))

    # export data as torch tensor
    self$x <- torch_tensor(x)
  },

  .getitem = function(i) {

    # find the rows in the data corresponding to the
    # correct site subset
    l <- which(self$sitename %in% self$sites[i])

    # here the true training data is
    # generated, subsetting data using the
    # index l, and selecting the correct columns
    # for training data x, and target data y
    list(
      x = self$x[l,],
      y = self$y[l]
    )
  },

  # return length of the number of
  # sites to do internal bookkeeping
  # on the data loader (batching?)
  .length = function() {
    length(self$sites)
  }
)
