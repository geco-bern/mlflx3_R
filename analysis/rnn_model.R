# model
library(torch)

rnn_model <- nn_module(
  initialize = function(
    input_size,
    hidden_size,
    conditional = 0,
    conditional_size,
    output_size = 1,
    num_layers = 2,
    dropout = 0.3
    ) {

    # self sets global variables
    # across functions after initialize
    self$conditional <- conditional

    # define the lstm section
    self$rnn <-nn_lstm(
        input_size = input_size,
        hidden_size = hidden_size,
        num_layers = num_layers,
        dropout = dropout,
        batch_first = TRUE
      )

    # conditional fully connected layer
    if (self$conditional) {
      self$fc1 <- nn_sequential(
        nn_linear(hidden_size + conditional_size, 64),
        nn_relu()
      )
    } else {
      self$fc1 <- nn_sequential(
        nn_linear(hidden_size, 64),
        nn_relu()
      )
    }

    # below connected sequential
    # layers can be collapsed into
    # one nn_sequential
    self$fc2 <- nn_sequential(
      nn_linear(64, 32),
      nn_relu()
    )

    self$fc3 <- nn_sequential(
      nn_linear(32, 16),
      nn_relu()
    )

    self$fc4 = nn_linear(16, output_size)
  },

  forward = function(x, c) {

    out <- self$rnn(x)[[1]] |> torch_squeeze()

    if (self$conditional) {
      out <- torch_cat(c(out, c), dim = 1)
    }

    y <- self$fc1(out) |>
      self$fc2()|>
      self$fc3()|>
      self$fc4() |>
      torch_squeeze()

    return(y)

  }
)
