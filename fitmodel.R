 library(keras)
 library(data.table)

 x_train=fread('hap_new.txt')
 y=x_train$outcome
 x_train$outcome=NULL

x_train=data.frame(x_train)
x_train=data.matrix(x_train)
y=as.factor(y)

penalty2=c(rep(1,17327),rep(1,4))

library(glmnet)


cv.lasso3 <- cv.glmnet(x_train, (y), family='binomial', alpha=0, parallel=TRUE, standardize=FALSE, penalty.factor=penalty2,type.measure='auc')
coeff=coef(cv.lasso3)
coef_name=rownames(coeff)
coeff=coeff[-c(1:3)]
coef_name=coef_name[-c(1:3)]
score=data.frame(coef_name,coeff)
fwrite(score,'score.csv')




# library(keras)
# library(data.table)
# 
# # x_train=fread('train_full.txt')
# # y_train=x_train$outcome
# 
# x_train_hap=fread('train_hap_22.txt')
# y_train_hap=x_train_hap$outcome
# 
# 
# # y_train[y_train=='Case']=1
# # y_train[y_train=='Control']=0
# # 
# # x_train$outcome=NULL
# # x_train=data.frame(x_train)
# 
# x_train_hap$outcome=NULL
# x_train_hap=data.frame(x_train_hap)
# 
# 
# x_train_hap=x_train_hap[1:78164,]
# y_train_hap=y_train_hap[1:78164]
# 
# get.gpu.count <- function() {
#   out <- system2("nvidia-smi", "-L", stdout=TRUE)
#   length(out)
# }
# 
# test_uk=fread('eur_crc_test.txt')
# y_uk=test_uk$outcome
# # test_uk$outcome=NULL
# # test_uk=data.frame(test_uk)
# # idx=match(colnames(x_train),colnames(test_uk))
# # test_uk=test_uk[,idx]
# # 
# 
# 
# 
# 
# #model <- keras_model_sequential()
# model1 <- keras_model_sequential()
# #model2 <- keras_model_sequential()
# #model3 <- keras_model_sequential()
# #model4 <- keras_model_sequential()
# 
# fscore=matrix(0,length(y_uk),1)
# 
# ###CNN 6 and 2 highest
# # x_train <- as.matrix(x_train)
# # dim(x_train) <- c(dim(x_train),1)
# # test_uk <- as.matrix(test_uk)
# # dim(test_uk) <- c(dim(test_uk),1)
# 
# 
# #model %>%
# #layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu',kernel_regularizer = regularizer_l2(0.0001),
# #               input_shape = c(ncol(x_train),1)) %>%
# #layer_max_pooling_1d(pool_size = 2) %>%
# #layer_flatten()%>%
# #layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# #layer_dropout(rate = 0.001) %>%
# #layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# #layer_dropout(rate = 0.001) %>%
# #layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# #layer_dropout(rate = 0.001) %>%
# #layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# #layer_dropout(rate = 0.001) %>%
# #layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# #layer_dropout(rate = 0.001) %>%
# #layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# #layer_dropout(rate = 0.001) %>%
# #layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# #layer_dropout(rate = 0.001) %>%
# #layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# #layer_dropout(rate = 0.001) %>%
# #layer_dense(units = 1, activation = 'sigmoid')
# 
# #### Model 1 AUC 0.5
# #model %>% 
# # layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu',kernel_regularizer = regularizer_l2(0.0001),
# #                input_shape = c(ncol(x_train),1)) %>% 
# #   layer_max_pooling_1d(pool_size = 2) %>%
# # layer_flatten()%>% 
# # layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# # layer_dropout(rate = 0.001) %>%
# # layer_dense(units = 1, activation = 'sigmoid') 
# 
# #### Model 2 AUC 0.626
# # model %>%
# #layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu',kernel_regularizer = regularizer_l2(0.0001),
# #input_shape = c(ncol(x_train),1)) %>%
# #layer_max_pooling_1d(pool_size = 2) %>%
# #layer_flatten()%>%
# #layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# #layer_dropout(rate = 0.001) %>%
# #layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# #layer_dropout(rate = 0.001) %>%
# #layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# #layer_dropout(rate = 0.001) %>%
# #layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# #layer_dropout(rate = 0.001) %>%
# #layer_dense(units = 1, activation = 'sigmoid')
# ##### Model 3
# #model3 %>% 
# #layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu',kernel_regularizer = regularizer_l2(0.0001),
# #               input_shape = c(ncol(x_train),1)) %>% 
# #layer_max_pooling_1d(pool_size = 2) %>%
# #layer_flatten()%>% 
# #layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# #layer_dropout(rate = 0.001) %>%
# #layer_dense(units = 1, activation = 'sigmoid') 
# ##### Model 4
# # model %>%
# #   layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu',kernel_regularizer = regularizer_l2(0.0001),
# #                 input_shape = c(ncol(x_train),1)) %>%
# #   # layer_max_pooling_1d(pool_size = 2) %>%
# #   layer_flatten()%>%
# #   layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# #   layer_dropout(rate = 0.001) %>%
# #   layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# #   layer_dropout(rate = 0.001) %>%
# #   layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
# #   layer_dropout(rate = 0.001) %>%
# #   layer_dense(units = 1, activation = 'sigmoid')
# # ####Model
# # parallel_model <- multi_gpu_model(model, gpus=get.gpu.count())
# # parallel_model  %>% compile(
# #   loss = 'binary_crossentropy',
# #   optimizer = optimizer_adam(lr=0.001),
# #   metrics = c('accuracy')
# # )
# # parallel_model %>% fit(
# #   x_train, y_train,
# #   epochs = 10,batch_size=7000,
# # )
# # score=parallel_model %>% predict(test_uk,batch_size=128)
# # 
# 
# test_uk_hap=fread('rpgeh_hap_22.txt')
# y_uk_hap=y_uk
# test_uk_hap=data.frame(test_uk_hap)
# idx=match(colnames(x_train_hap),colnames(test_uk_hap))
# test_uk_hap=test_uk_hap[,idx]
# 
# x_train <- as.matrix(x_train_hap)
# dim(x_train) <- c(dim(x_train),1)
# test_uk <- as.matrix(test_uk_hap)
# dim(test_uk) <- c(dim(test_uk),1)
# 
# model1 %>%
#   layer_conv_1d(filters = 64, kernel_size = 1, activation = 'relu',kernel_regularizer = regularizer_l2(0.0001),
#                 input_shape = c(ncol(x_train),1)) %>%
#   # layer_max_pooling_1d(pool_size = 2) %>%
#   layer_flatten()%>%
#   layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
#   layer_dropout(rate = 0.001) %>%
#   layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
#   layer_dropout(rate = 0.001) %>%
#   layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
#   layer_dropout(rate = 0.001) %>%
#   layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
#   layer_dropout(rate = 0.001) %>%
#   layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.0001), activation = 'relu') %>%
#   layer_dropout(rate = 0.001) %>%
#   layer_dense(units = 1, activation = 'sigmoid')
# ###Model
# parallel_model <- multi_gpu_model(model1, gpus=get.gpu.count())
# parallel_model  %>% compile(
#   loss = 'binary_crossentropy',
#   optimizer = optimizer_adam(lr=0.001),
#   metrics = c('accuracy')
# )
# parallel_model %>% fit(
#   x_train, y_train_hap,
#   epochs = 10,batch_size=7000,
# )
# score1=parallel_model %>% predict(test_uk,batch_size=128)
# 
# fwrite(data.frame(score1,y_uk_hap),'score.csv')
