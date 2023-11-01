#' LSTM model
#'
#' @param input_size input size (nr of columns with time series drivers)
#' @param hidden_size the size of the LSTM hidden layers in the module
#' @param output_size the output size of the model
#' @param num_layers number of layers in the LSTM module
#' @param dropout dropout rate of the LSTM module
#'
#' @return a bare LSTM model, to be configured and run
#' @export

rnn_model <- torch::nn_module(
  "Net",
  initialize = function(
    input_size,
    hidden_size,
    output_size,
    num_layers = 2,
    dropout = 0.3
  ) {

    # define the lstm section
    self$lstm <-torch::nn_lstm(
      input_size = input_size,
      hidden_size = hidden_size,
      num_layers = num_layers,
      dropout = dropout,
      batch_first = TRUE
    )

    # The fully connected standard neural
    # net (can be split into sections using
    # nn_sequential() as in the original python
    # code, but this is cleaner)
    self$fc <-torch::nn_sequential(
      torch::nn_linear(hidden_size, 64),
      torch::nn_relu(),
      torch::nn_linear(64, 32),
      torch::nn_relu(),
      torch::nn_linear(32, 16),
      torch::nn_relu(),
      torch::nn_linear(16, output_size)
    )
  },

  forward = function(x) {

    # run the lstm squeeze out
    # the additional unity dimension
    out <- self$lstm(x)[[1]]

    # stack the normal neural net
    # components on the lstm features
    # and squeeze the data to comform
    # to the target output shape
    self$fc(out) |>
      torch::torch_squeeze(-1)
  }
)
