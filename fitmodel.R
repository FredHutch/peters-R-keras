library(keras)

library(data.table)

x_train=fread('x_train.csv')

x_train=data.frame(x_train)

y_train=x_train[,1]

x_train=data.matrix(x_train[,-c(1:3)])
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
test_uk=scale(test_uk[,1:30000])
###########test3

#test_plco=fread('test_plco.csv')

#test_plco=data.frame(test_plco)

#y_plco=test_plco[,1]

#test_plco=data.matrix(test_plco[,-1])


model <- keras_model_sequential()

model %>%

  layer_dense(units = 3000, activation = 'relu', input_shape = c(5000)) %>%

  layer_dropout(rate = 0.5) %>%

  layer_dense(units = 2000, activation = 'relu') %>%

  layer_dropout(rate = 0.5) %>%

  layer_dense(units = 1000, activation = 'relu') %>%

  layer_dropout(rate = 0.5) %>%

  layer_dense(units = 500, activation = 'relu') %>%

  layer_dropout(rate = 0.5) %>%
  
  layer_dense(units = 1, activation = 'sigmoid')

 

parallel_model <- multi_gpu_model(model, gpus=get.gpu.count())

parallel_model %>%   compile(

    loss = 'binary_crossentropy',

    optimizer = 'rmsprop',

    metrics = c('accuracy')

  )

#score1=c()


#for(i in c(20,40,60,80,100,120,140,160)){
parallel_model %>% fit(scale(x_train[,1:5000]), y_train, epochs = 20, batch_size = 258)
#score1 = c(score1,parallel_model %>% evaluate(x_train, y_train, batch_size=128))
#score = parallel_model %>% evaluate(x_test, y_test, batch_size=128)
score = parallel_model %>% predict(scale(test_uk[,1:5000]), batch_size=128)
#score1 = parallel_model %>% predict_classes(test_uk, y_uk, batch_size=258)
#score1 = predict_classes(parallel_model,test_uk, batch_size=258)
#score2 = parallel_model %>% evaluate(test_plco, y_plco, batch_size=128)
# }
#ss=cbind(score,score1,score2)
#score = parallel_model %>% predict(x_test, batch_size=128)

fwrite(data.frame(score,y_uk),'score.csv')

