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

model <- keras_model_sequential()
fscore=matrix(0,length(y_uk),1)
model %>%
 layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.001), activation = 'relu', input_shape = c(ncol(x_train))) %>%
 layer_dropout(rate = 0.2) %>%
 layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.001), activation = 'relu') %>%
 layer_dropout(rate = 0.2) %>%
 layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.001), activation = 'relu') %>%
layer_dropout(rate = 0.2) %>%
 layer_dense(units = 320, kernel_regularizer = regularizer_l2(0.001), activation = 'relu') %>%
 layer_dropout(rate = 0.2) %>%
 layer_dense(units = 12, kernel_regularizer = regularizer_l2(0.001), activation = 'relu') %>%
 layer_dropout(rate = 0.2) %>%
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

parallel_model <- multi_gpu_model(model, gpus=get.gpu.count())


parallel_model  %>% compile(
  loss = 'binary_crossentropy',
 #optimizer = optimizer_rmsprop(lr=0.001),
 optimizer = optimizer_adam(lr=0.001),
 metrics = c('accuracy')
 #metrics = c(metric_auc)
)
#score1=c()
#filepath <- "model_reg.hdf5" # set up your own filepath
#checkpoint <- callback_model_checkpoint(filepath = filepath, monitor = "val_acc", verbose = 1,
#                                       save_best_only = TRUE,
#                                       save_weights_only = FALSE, mode = "auto")
#reduce_lr <- callback_reduce_lr_on_plateau(monitor = "val_acc", factor = 0.9,
#                                          patience = 20, verbose = 1, mode = "auto",
#                                          min_lr = 0.0001)

history.reg <- parallel_model %>% fit(
x_train, y_train,
epochs = 5,batch_size=50000
)

#pdf('history.reg.pdf')
#plot(history.reg)
#dev.off()
#max(history.reg$metrics$val_acc)
# load and evaluate best model
#rm(parallel_model)
#model.reg <- keras:::keras$models$load_model(filepath)
#score=model.reg %>% predict(test_uk,batch_size=nrow(test_uk))
#score=model.reg %>% predict_proba(test_uk)
score = parallel_model %>% predict(test_uk,batch_size=128)
#score = parallel_model %>% predict(test_uk,batch_size=500)
#fscore=cbind(fscore,score)
#}
fwrite(data.frame(score,y_uk),'score.csv')

