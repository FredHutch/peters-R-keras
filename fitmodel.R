library(keras)

library(data.table)

x_train=fread('x_train.csv')

x_train=data.frame(x_train)

y_train=x_train[,1]

x_train=data.matrix(x_train[,-1])
x_train=x_train[,1:30000]

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
test_uk=data.matrix(test_uk[,-1])
###########test3

#test_plco=fread('test_plco.csv')

#test_plco=data.frame(test_plco)

#y_plco=test_plco[,1]

#test_plco=data.matrix(test_plco[,-1])


model <- keras_model_sequential()

model %>%

  layer_dense(units = 1024, activation = 'relu', input_shape = c(30000)) %>%

  layer_dropout(rate = 0.5) %>%

  layer_dense(units = 400, activation = 'relu') %>%

  layer_dropout(rate = 0.3) %>%

  layer_dense(units = 1, activation = 'sigmoid')

 

parallel_model <- multi_gpu_model(model, gpus=8)

parallel_model %>%   compile(

    loss = 'binary_crossentropy',

    optimizer = 'rmsprop',

    metrics = c('accuracy')

  )

#score1=c()


#for(i in c(20,40,60,80,100,120,140,160)){
parallel_model %>% fit(x_train, y_train, epochs = 20, batch_size = 128)
#score1 = c(score1,parallel_model %>% evaluate(x_train, y_train, batch_size=128))
#score = parallel_model %>% evaluate(x_test, y_test, batch_size=128)
#score = parallel_model %>% predict(x_test, batch_size=128)
score1 = parallel_model %>% evaluate(test_uk, y_uk, batch_size=128)
#score2 = parallel_model %>% evaluate(test_plco, y_plco, batch_size=128)
# }
#ss=cbind(score,score1,score2)
#score = parallel_model %>% predict(x_test, batch_size=128)
fwrite(data.frame(score1),'score.csv')

