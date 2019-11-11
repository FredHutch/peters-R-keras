library(keras)
library(data.table)

x_train=fread('train_gwass.txt')
y_train=x_train$outcome
x_train$outcome=NULL
x_train=data.frame(x_train)


get.gpu.count <- function() {
  out <- system2("nvidia-smi", "-L", stdout=TRUE)
  length(out)
}

test_uk=fread('eur_crc_test.txt')
y_uk=test_uk$outcome
test_uk$outcome=NULL
test_uk=data.frame(test_uk)

parallel_model <- keras_model_sequential()
fscore=matrix(0,length(y_uk),1)
parallel_model %>%
 layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu', input_shape = c(ncol(x_train))) %>%
 layer_dropout(rate = 0.001) %>%
 layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
 layer_dropout(rate = 0.001) %>%
layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
layer_dropout(rate = 0.001) %>%
 layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
layer_dropout(rate = 0.001) %>%
 layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
 layer_dropout(rate = 0.001) %>%
 layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
 layer_dropout(rate = 0.001) %>%
 layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
 layer_dropout(rate = 0.001) %>%
layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
 layer_dropout(rate = 0.001) %>%
 layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
 layer_dropout(rate = 0.001) %>%
 layer_dense(units = 1, activation = 'sigmoid') 

###CNN
x_train <- as.matrix(x_train)
#dim(x_train) <- c(dim(x_train),1)
test_uk <- as.matrix(test_uk)
#dim(test_uk) <- c(dim(test_uk),1)
#model %>% 
# layer_conv_1d(filters = 64, kernel_size = 3, activation = 'relu',
#                input_shape = c(16485,1)) %>% 
#  layer_conv_1d(filters = 64, kernel_size = 3, activation = 'relu') %>% 
#  layer_max_pooling_1d(pool_size = 3) %>% 
 # layer_conv_1d(filters = 128, kernel_size = 3, activation = 'relu') %>% 
#  layer_conv_1d(filters = 128, kernel_size = 3, activation = 'relu') %>% 
#  layer_global_average_pooling_1d() %>% 
#  layer_dropout(rate = 0.5) %>% 
#  layer_dense(units = 1, activation = 'sigmoid') 

#parallel_model <- multi_gpu_model(model, gpus=get.gpu.count())


parallel_model  %>% compile(
  loss = 'binary_crossentropy',
 #optimizer = optimizer_rmsprop(lr=0.001),
 optimizer = optimizer_adam(lr=0.001),
 metrics = c('accuracy')
 #metrics = c(metric_auc)
)
#score1=c()


checkpoint_dir <- "checkpoints"
dir.create(checkpoint_dir, showWarnings = FALSE)
filepath <- file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")


checkpoint <- callback_model_checkpoint(filepath = filepath, monitor = "val_acc", verbose = 1,
                                       save_best_only = TRUE,
                                       save_weights_only = TRUE, mode = "auto")
#reduce_lr <- callback_reduce_lr_on_plateau(monitor = "val_acc", factor = 0.9,
#                                          patience = 20, verbose = 1, mode = "auto",
#                                          min_lr = 0.0001)

history.reg <- parallel_model %>% fit(
x_train, y_train,
epochs = 10,batch_size=10000,validation_data = list(test_uk,y_uk), callbacks = list(checkpoint)
)
#save_model_weights_hdf5(parallel_model,filepath)
#fresh_model <- load_model_weights_hdf5(filepath, by_name = TRUE)
score = parallel_model %>% predict(test_uk,batch_size=128)
score2 = parallel_model %>% predict(x_train,batch_size=128)
list.files(checkpoint_dir)
create_model <- function() {
  model1 <- keras_model_sequential() %>%
    layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu', input_shape = c(ncol(x_train))) %>%
 layer_dropout(rate = 0.001) %>%
 layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
 layer_dropout(rate = 0.001) %>%
 layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
layer_dropout(rate = 0.001) %>%
 layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
 layer_dropout(rate = 0.001) %>%
   layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
 layer_dropout(rate = 0.001) %>%
   layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
 layer_dropout(rate = 0.001) %>%
   layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
 layer_dropout(rate = 0.001) %>%
  layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
 layer_dropout(rate = 0.001) %>%
 layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
 layer_dropout(rate = 0.001) %>%
 layer_dense(units = 1, activation = 'sigmoid')
  model1 %>% compile(
    loss = 'binary_crossentropy',
 #optimizer = optimizer_rmsprop(lr=0.001),
 optimizer = optimizer_adam(lr=0.001),
 metrics = c('accuracy')
 #metrics = c(metric_auc)
)
  model1
}

fresh_model <-  create_model()
fresh_model %>% load_model_weights_hdf5(
  file.path(checkpoint_dir, list.files(checkpoint_dir)[length(list.files(checkpoint_dir))])
)

score1 = fresh_model %>% predict(test_uk,batch_size=128)

fwrite(data.frame(score1,score,score2,y_uk),'score.csv')
fwrite(data.frame(score2,y_train),'score2.csv')
