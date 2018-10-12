library(keras)

library(data.table)

x_train=fread('x_train.csv')

x_train=data.frame(x_train)

y_train=x_train[,1]

x_train=data.matrix(x_train[,-c(1:3)])
#x_train=data.matrix(x_train[,-c(1)])
x_train=scale(x_train[,1:30000])

get.gpu.count <- function() {
  out <- system2("nvidia-smi", "-L", stdout=TRUE)
  length(out)
}

######test1
#x_test=fread('x_test.csv')

#x_test=data.frame(x_test)

#y_test=x_test[,1]


#x_test=data.matrix(x_test[,-1])
#x_test=x_test[,1:30000]

########test2
test_uk=fread('test_uk.csv')

test_uk=data.frame(test_uk)

y_uk=test_uk[,1]
test_uk=data.matrix(test_uk[,-c(1:3)])
#test_uk=data.matrix(test_uk[,-c(1)])
test_uk=scale(test_uk[,1:30000])
###########test3
x_train=scale(x_train[,1:30000])
test_uk=scale(test_uk[,1:30000])
#test_plco=fread('test_plco.csv')

#test_plco=data.frame(test_plco)

#y_plco=test_plco[,1]

#test_plco=data.matrix(test_plco[,-1])

#act=c(10,100,1000,2000)
model <- keras_model_sequential()
fscore=matrix(0,length(y_uk),1)
#for(i in 1:4){
model %>%
  layer_dense(units = 15000, kernel_regularizer = regularizer_l2(0.01), activation = 'tanh', input_shape = c(30000)) %>%
  layer_dropout(rate = 0) %>%
  #layer_dense(units = 150, kernel_regularizer = regularizer_l2(0.001), activation = 'sigmoid') %>%
  #layer_dropout(rate = 0.3) %>%
  #layer_dense(units = 150, kernel_regularizer = regularizer_l2(0.001), activation = 'sigmoid') %>%
  #layer_dropout(rate = 0.3) %>%
  #layer_dense(units = 150, kernel_regularizer = regularizer_l2(0.001), activation = 'sigmoid') %>%
  #layer_dropout(rate = 0.3) %>%
  #layer_dense(units = 150, kernel_regularizer = regularizer_l2(0.001), activation = 'sigmoid') %>%
  #layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = 'sigmoid') 

parallel_model <- multi_gpu_model(model, gpus=get.gpu.count())

parallel_model  %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(lr=0.001),
  metrics = c('accuracy')
)
#score1=c()
#filepath <- "model_reg.hdf5" # set up your own filepath
#checkpoint <- callback_model_checkpoint(filepath = filepath, monitor = "val_acc", verbose = 1,
#                                       save_best_only = TRUE,
#                                       save_weights_only = FALSE, mode = "auto")
#reduce_lr <- callback_reduce_lr_on_plateau(monitor = "val_acc", factor = 0.9,
#                                          patience = 20, verbose = 1, mode = "auto",
#                                          min_lr = 0.00001)

#history.reg <- parallel_model %>% fit(
#x_train, y_train,
#epochs = 10, batch_size = nrow(x_train),
#validation_data = list(test_uk, y_uk), shuffle = FALSE,
#callbacks = list(checkpoint, reduce_lr)
#)
parallel_model %>% fit(x_train, y_train, epochs = 10, batch_size = nrow(x_train))
# plot training loss and accuracy
#pdf('history.reg.pdf')
#plot(history.reg)
#dev.off()
#max(history.reg$metrics$val_acc)
# load and evaluate best model
#rm(parallel_model)
#model.reg <- keras:::keras$models$load_model(filepath)
#score=model.reg %>% predict(test_uk,batch_size=nrow(test_uk))
#score=model.reg %>% predict_proba(test_uk)
score = parallel_model %>% predict(test_uk,batch_size=nrow(test_uk))
#fscore=cbind(fscore,score)
#}
fwrite(data.frame(score,y_uk),'score.csv')

