# Embeddings instead of one-hot encoding
# so the embedding works as such:
# cardinalities are the number of classes
# per column (factor variable). These settings
# are passed to initialized and construct the
# embedding framework - or embedding module.
#
# Cardinalities are computed up front on the
# total dataset (to cover all classes). The forward
# function defines the processing of the data with
# self$embeddings (on x - which is true data)
#
# In your full model you then call the embedding_module
# with the cardinality settings (assigned to self),
# this function is then called upon construction / initializing
# and is then used in the forward part of the model
# which feeds in real data, concats the values where
# necessary

embedding_module <- nn_module(

  initialize = function(cardinalities) {
    self$embeddings = nn_module_list(
      lapply(
        cardinalities,
        function(x) nn_embedding(num_embeddings = x, embedding_dim = ceiling(x/2)))
      )
  },

  forward = function(x) {
    embedded <- vector(mode = "list", length = length(self$embeddings))
    for (i in 1:length(self$embeddings)) {
      embedded[[i]] <- self$embeddings[[i]](x[ , i])
    }
    torch_cat(embedded, dim = 2)
  }
)

net <- nn_module(
  "mushroom_net",

  initialize = function(cardinalities,
                        num_numerical,
                        fc1_dim,
                        fc2_dim) {
    self$embedder <- embedding_module(cardinalities)
    self$fc1 <- nn_linear(
      sum(map(cardinalities, function(x) ceiling(x/2)) %>% unlist()) # automatically set dimensions right
      + num_numerical, fc1_dim)
    self$fc2 <- nn_linear(fc1_dim, fc2_dim)
    self$output <- nn_linear(fc2_dim, 1)
  },

  forward = function(xcat, xnum) {
    embedded <- self$embedder(xcat)
    all <- torch_cat(list(embedded, xnum$to(dtype = torch_float())), dim = 2)
    all %>% self$fc1() %>%
      nnf_relu() %>%
      self$fc2() %>%
      self$output() %>%
      nnf_sigmoid()
  }
)
