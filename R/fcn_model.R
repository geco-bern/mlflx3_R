#' FCN model
#'
#' @param input_size input size (nr of columns with time series drivers)
#' @param output_size the output size of the model
#'
#' @return a bare FCN (DNN) model, to be configured and run
#' @export

rnn_model <- torch::nn_module(
  "FCN",
  initialize = function(
    input_size,
    output_size
  ) {

    # The fully connected standard neural
    # net (can be split into sections using
    # nn_sequential() as in the original python
    # code, but this is cleaner)
    self$fc <-torch::nn_sequential(
      torch::nn_linear(input_size, 64),
      torch::nn_relu(),
      torch::nn_linear(64, 32),
      torch::nn_relu(),
      torch::nn_linear(32, 16),
      torch::nn_relu(),
      torch::nn_linear(16, output_size)
    )
  },

  forward = function(x) {

    # stack the normal neural net
    # components on the lstm features
    # and squeeze the data to comform
    # to the target output shape
    self$fc(x) |>
      torch::torch_squeeze(-1)
  }
)
