
gpp_mean <- mean(gpp_train)
gpp_sd <- sd(gpp_train)

gpp_dataset <- dataset(
  name = "gpp_dataset",

  initialize = function(
    x,
    n_timesteps,
    n_forecast,
    sample_frac = 1
    ) {

    self$n_timesteps <- n_timesteps
    self$n_forecast <- n_forecast

    # normalize columns in input matrix


    # convert to tensor
    self$x <- torch_tensor((x - train_mean) / train_sd)

    # split out samples
    n <- length(self$x) - self$n_timesteps - self$n_forecast + 1

    self$starts <- sort(sample.int(
      n = n,
      size = n * sample_frac
    ))

  },

  .getitem = function(i) {

    start <- self$starts[i]
    end <- start + self$n_timesteps - 1
    pred_length <- self$n_forecast

    list(
      x = self$x[start:end],
      y = self$x[(end + 1):(end + pred_length)]$squeeze(2)
    )

  },

  .length = function() {
    length(self$starts)
  }
)
