# LSTM model

rnn_model <- nn_module(
  "Net",
  initialize = function(
    input_size,
    hidden_size,
    output_size,
    num_layers = 2,
    dropout = 0.3
    ) {

    # define the lstm section
    self$lstm <-nn_lstm(
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
    self$fc <-nn_sequential(
        nn_linear(hidden_size, 64),
        nn_relu(),
        nn_linear(64, 32),
        nn_relu(),
        nn_linear(32, 16),
        nn_relu(),
        nn_linear(16, output_size)
    )
  },

  forward = function(x, c) {

    # run the lstm squeeze out
    # the additional unity dimension
    out <- self$lstm(x)[[1]]

    # stack the normal neural net
    # components on the lstm features
    # and squeeze the data to comform
    # to the target output shape
    self$fc(out) |>
      torch_squeeze(-1)
  }
)
