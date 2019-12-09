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

parallel_model <- keras_model_sequential()
fscore=matrix(0,length(y_uk),1)

###CNN
x_train <- as.matrix(x_train)
dim(x_train) <- c(dim(x_train),1)
test_uk <- as.matrix(test_uk)
dim(test_uk) <- c(dim(test_uk),1)
parallel_model %>% 
 layer_conv_1d(filters = 64, kernel_size = 3, activation = 'relu',
                input_shape = c(ncol(x_train),1)) %>% 
  layer_conv_1d(filters = 64, kernel_size = 3, activation = 'relu') %>% 
  layer_max_pooling_1d(pool_size = 3) %>% 
 layer_conv_1d(filters = 128, kernel_size = 3, activation = 'relu') %>% 
  layer_conv_1d(filters = 128, kernel_size = 3, activation = 'relu') %>% 
  layer_global_average_pooling_1d() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid') 

#parallel_model <- multi_gpu_model(model, gpus=get.gpu.count())

parallel_model  %>% compile(
 loss = 'binary_crossentropy',
 #optimizer = optimizer_rmsprop(lr=0.001),
 optimizer = optimizer_adam(lr=0.001),
 metrics = c('accuracy')
 #metrics = c(metric_auc)
)



 parallel_model %>% fit(
x_train, y_train,
epochs = 10,batch_size=128,
)

  score=parallel_model %>% predict(test_uk,batch_size=128)
 


fwrite(data.frame(score,y_uk),'score.csv')

