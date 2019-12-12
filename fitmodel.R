library(keras)
library(data.table)

x_train=fread('train_full.txt')
y_train=x_train$outcome

y_train[y_train=='Case']=1
y_train[y_train=='Control']=0

x_train$outcome=NULL
x_train=data.frame(x_train)
#x_train=x_train[1:30000,]
#y_train=y_train[1:30000]

get.gpu.count <- function() {
  out <- system2("nvidia-smi", "-L", stdout=TRUE)
  length(out)
}

test_uk=fread('eur_crc_test.txt')
y_uk=test_uk$outcome
test_uk$outcome=NULL
test_uk=data.frame(test_uk)

model <- keras_model_sequential()
#model1 <- keras_model_sequential()
#model2 <- keras_model_sequential()
#model3 <- keras_model_sequential()
#model4 <- keras_model_sequential()

fscore=matrix(0,length(y_uk),1)

###CNN
x_train <- as.matrix(x_train)
dim(x_train) <- c(dim(x_train),1)
test_uk <- as.matrix(test_uk)
dim(test_uk) <- c(dim(test_uk),1)
model %>%
layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu',kernel_regularizer = regularizer_l2(0.0001),
               input_shape = c(ncol(x_train),1)) %>%
layer_max_pooling_1d(pool_size = 2) %>%
layer_flatten()%>%
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
layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
layer_dropout(rate = 0.001) %>%
layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
layer_dropout(rate = 0.001) %>%
layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
layer_dropout(rate = 0.001) %>%
layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
layer_dropout(rate = 0.001) %>%
layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
layer_dropout(rate = 0.001) %>%
layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
layer_dropout(rate = 0.001) %>%
layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
layer_dropout(rate = 0.001) %>%
layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
layer_dropout(rate = 0.001) %>%
layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
layer_dropout(rate = 0.001) %>%
layer_dense(units = 1, activation = 'sigmoid')

#### Model 1 AUC 0.5
# model1 %>% 
#   layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu',kernel_regularizer = regularizer_l2(0.0001),
#                 input_shape = c(ncol(x_train),1)) %>% 
#   layer_max_pooling_1d(pool_size = 2) %>%
#   layer_flatten()%>% 
#   layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
#   layer_dropout(rate = 0.001) %>%
#   layer_dense(units = 1, activation = 'sigmoid') 

#### Model 2 AUC 0.626
# model2 %>%
# layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu',kernel_regularizer = regularizer_l2(0.0001),
# input_shape = c(ncol(x_train),1)) %>%
# layer_max_pooling_1d(pool_size = 2) %>%
# layer_flatten()%>%
# layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# layer_dropout(rate = 0.001) %>%
# layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# layer_dropout(rate = 0.001) %>%
# layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# layer_dropout(rate = 0.001) %>%
# layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# layer_dropout(rate = 0.001) %>%
# layer_dense(units = 1, activation = 'sigmoid')
##### Model 3
#model3 %>% 
#layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu',kernel_regularizer = regularizer_l2(0.0001),
#               input_shape = c(ncol(x_train),1)) %>% 
#layer_max_pooling_1d(pool_size = 2) %>%
#layer_flatten()%>% 
#layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
#layer_dropout(rate = 0.001) %>%
#layer_dense(units = 1, activation = 'sigmoid') 
##### Model 4
# model4 %>%
# layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu',kernel_regularizer = regularizer_l2(0.0001),
#                input_shape = c(ncol(x_train),1)) %>%
# layer_max_pooling_1d(pool_size = 2) %>%
# layer_flatten()%>%
# layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# layer_dropout(rate = 0.001) %>%
# layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# layer_dropout(rate = 0.001) %>%
# layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# layer_dropout(rate = 0.001) %>%
# layer_dense(units = 1, activation = 'sigmoid')
####Model
parallel_model <- multi_gpu_model(model, gpus=get.gpu.count())
parallel_model  %>% compile(
loss = 'binary_crossentropy',
optimizer = optimizer_rmsprop(lr=0.001),
optimizer = optimizer_adam(lr=0.001),
metrics = c('accuracy')
metrics = c(metric_auc)
)
parallel_model %>% fit(
  x_train, y_train,
  epochs = 10,batch_size=5000,
)
score=parallel_model %>% predict(test_uk,batch_size=128)
######### Model 1
# parallel_model <- multi_gpu_model(model1, gpus=get.gpu.count())
# parallel_model  %>% compile(
#   loss = 'binary_crossentropy',
#   #optimizer = optimizer_rmsprop(lr=0.001),
#   optimizer = optimizer_adam(lr=0.001),
#   metrics = c('accuracy')
#   #metrics = c(metric_auc)
# )
# parallel_model %>% fit(
#   x_train, y_train,
#   epochs = 10,batch_size=5000,
# )
# score1=parallel_model %>% predict(test_uk,batch_size=128)
# ######### Model 2
# parallel_model <- multi_gpu_model(model2, gpus=get.gpu.count())
# parallel_model  %>% compile(
#   loss = 'binary_crossentropy',
#   #optimizer = optimizer_rmsprop(lr=0.001),
#   optimizer = optimizer_adam(lr=0.001),
#   metrics = c('accuracy')
#   #metrics = c(metric_auc)
# )
# parallel_model %>% fit(
#   x_train, y_train,
#   epochs = 10,batch_size=5000,
# )
# score2=parallel_model %>% predict(test_uk,batch_size=128)
######
######### Model 3
# parallel_model <- multi_gpu_model(model3, gpus=get.gpu.count())
# parallel_model  %>% compile(
#   loss = 'binary_crossentropy',
#   #optimizer = optimizer_rmsprop(lr=0.001),
#   optimizer = optimizer_adam(lr=0.001),
#   metrics = c('accuracy')
#   #metrics = c(metric_auc)
# )
# parallel_model %>% fit(
#   x_train, y_train,
#   epochs = 10,batch_size=5000,
# )
# score3=parallel_model %>% predict(test_uk,batch_size=128)
######### Model 4
# parallel_model <- multi_gpu_model(model4, gpus=get.gpu.count())
# parallel_model  %>% compile(
#   loss = 'binary_crossentropy',
#   #optimizer = optimizer_rmsprop(lr=0.001),
#   optimizer = optimizer_adam(lr=0.001),
#   metrics = c('accuracy')
#   #metrics = c(metric_auc)
# )
# parallel_model %>% fit(
#   x_train, y_train,
#   epochs = 10,batch_size=5000,
# )
# score4=parallel_model %>% predict(test_uk,batch_size=128)

fwrite(data.frame(score,y_uk),'score.csv')

