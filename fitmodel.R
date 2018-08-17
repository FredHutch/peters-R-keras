library(keras)

library(data.table)

x_train=fread('x_train.csv')

x_train=data.frame(x_train)

y_train=x_train[,1]

x_train=data.matrix(x_train[,-1])

x_test=fread('x_test.csv')

x_test=data.frame(x_test)

y_test=x_test[,1]

x_test=data.matrix(x_test[,-1])


model <- keras_model_sequential()

model %>%

  layer_dense(units = 1024, activation = 'relu', input_shape = c(99983)) %>%

  layer_dropout(rate = 0.5) %>%

  layer_dense(units = 664, activation = 'relu') %>%

  layer_dropout(rate = 0.5) %>%

  layer_dense(units = 1, activation = 'sigmoid')

 

parallel_model <- multi_gpu_model(model, gpus=8)

parallel_model %>%   compile(

    loss = 'binary_crossentropy',

    optimizer = 'rmsprop',

    metrics = c('accuracy')

  )

#score1=c()
#score=c()

#for(i in c(20,40,60,80,100,120,140,160)){
parallel_model %>% fit(x_train, y_train, epochs = 20, batch_size = 128)
#score1 = c(score1,parallel_model %>% evaluate(x_train, y_train, batch_size=128))
score = parallel_model %>% evaluate(x_test, y_test, batch_size=128)
# }
#ss=cbind(score1,score)
#score = parallel_model %>% predict(x_test, batch_size=128)
fwrite(data.frame(score),'score.csv')

