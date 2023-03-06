#load library
library(ggplot2)
library(lmtest)
library(glmnet)
library(MASS)
library(faraway)

################## Data Pre processing ####################
#Load Data
data_v1 = read.csv("reduced_data.csv")
head(data_v1)
# missing value: 239 Missing Value in price
sum(is.na(data_v1))
# drop NAH 
data_nomissing = na.omit(data_v1)
#check omit work or not
sum(is.na(data_nomissing))

# Drop Columns
drop <- c("id","datetime" , "timezone" , "timestamp", "source" ,"destination" , "short_summary" , "latitude" , "longitude", "product_id" , "short_summary", "long_summary", "humidity" , "windGust" , "windGustTime" , "icon", "dewPoint" , "pressure" , "windBearing" , "cloudCover" , "ozone" , "moonPhase")
data_afterdrop = data_nomissing[,!(names(data_nomissing) %in% drop)]
head(data_afterdrop)

# to factors
data_afterdrop$cab_type <- as.factor(data_afterdrop$cab_type)
data_afterdrop$name <- as.factor(data_afterdrop$name)
head(data_afterdrop)

#Check the response value distribution
hist(data_afterdrop$price, col = "cyan3")

#Transform data 
data_afterdrop$price = log(data_afterdrop$price)

#histogram after transformation
hist(data_afterdrop$price, col = "cyan3")

#balanced of the Lyft and Uber 
ggplot(data = data_afterdrop, aes(cab_type,fill = cab_type)) + geom_bar()
ggplot(data = data_afterdrop, aes(name, fill = cab_type)) + geom_bar()  


#Relaitonship between distance and price, and its correlation
m1=lm(price~distance,data = data_afterdrop)
cor(data_afterdrop$distance, data_afterdrop$price)
plot(data_afterdrop$distance,data_afterdrop$price,pch=1,col="cyan4")
abline(coef=m1$coefficients,c="orange",lwd=3)

##################### Models ##########################

#full model regression
lm_full_v1 = lm(price ~. , data = data_afterdrop)
summary(lm_full_v1)

#Cab_type Interaction 
lm_full = lm(price ~. + name:(.), data = data_afterdrop)
summary(lm_full)

# with interaction, the R2 become larger, from 0.93 to 0.94. 
anova(lm_full_v1, lm_full)

#Surge multiplier
lm_multiplier = lm( surge_multiplier ~. -price ,data = data_afterdrop)
summary(lm_multiplier)

#Without those non_significant predictors, R2 performance
lm_multiplier_without_nonsig =  lm(surge_multiplier  ~ name + precipIntensity + precipProbability +  apparentTemperatureHighTime + apparentTemperatureMinTime, data = data_afterdrop)
summary(lm_multiplier_without_nonsig)

#Model Assumptions
plot(lm_full)
bptest(lm_full)
shapiro.test(resid(lm_full))

############# Model Selections (Backward / Forward) ##########

#backward step model using AIC
fit_back_aic = step(lm_full, direction = "backward", trace = 0)

#null model
lm_for_null = lm(price ~ 1 , data = data_afterdrop)

#forward step model using AIC

lm_for_aic = step(lm_full,direction = "forward",
                  trace = 0)

#Stepwise Both Direction Model using AIC
lm_both_aic = step(lm_full,
                   direction = "both",
                   trace = 0)

#Compare models for AIC group
AIC(fit_back_aic , lm_for_aic , lm_both_aic)
summary(fit_back_aic)$adj.r.squared
summary(lm_for_aic)$adj.r.squared
summary(lm_both_aic)$adj.r.squared

#Model Assumptions for best of three
plot(fit_back_aic, pch = 1, cex = 0.5)
bptest(fit_back_aic)
shapiro.test(resid(fit_back_aic))

#using BIC as penalty
n = nrow(data_afterdrop)

#backward step model using BIC
fit_back_bic = step(lm_full, direction = "backward", trace = 0, k = log(n) )

# null model
lm_for_null = lm(price ~ 1 , data = data_afterdrop)

#forward step model using BIC
lm_for_bic = step(lm_full,
                  direction = "forward",
                  trace = 0, k = log(n) )

#Stepwise Both Direction Model using BIC
lm_both_bic = step(lm_full,
                   trace = 0,
                   k = log(n) )

#Model Comparison
BIC(fit_back_bic, lm_for_bic , lm_both_bic)

#adj r2
summary(fit_back_bic)$adj.r.squared
summary(lm_for_bic)$adj.r.squared
summary(lm_both_bic)$adj.r.squared

#Collinearity
vif(lm_both_aic)
vif(lm_both_bic)

#Model Assumptions
plot(fit_back_bic)
bptest(fit_back_bic)
shapiro.test(resid(fit_back_bic))

#The last obs is an influential point
out_i = which(cooks.distance(fit_back_aic) > 4 / length(cooks.distance(fit_back_aic)))
data_new = data_afterdrop[-out_i,]

#refit the both and check model assumption
lm_both_aic_without_inf = lm(price ~ day + month + name + distance + surge_multiplier + 
                               visibility + temperatureHighTime + temperatureLowTime + apparentTemperatureHigh + 
                               apparentTemperatureLow + temperatureMin + temperatureMinTime + apparentTemperatureMinTime, 
                             data = data_new)

#Model Assumpiton check again
plot(lm_both_aic_without_inf, cex = 0.5)
bptest(lm_both_aic_without_inf)
shapiro.test(resid(lm_both_aic_without_inf))

n = nrow(data_new)

#Create k equally size folds
k = 5
folds <- cut(1:n,breaks=k,labels=FALSE)

RMSE_kcv_both_aic = RMSE_kcv_both_bic = numeric(k)

#Perform a k-fold cross validation
for(i in 1:k)
{
  # Find the indices for test data
  test_index = which(folds==i)
  
  # Obtain training/test data
  test_data = data_new[test_index, ]
  training_data = data_new[-test_index, ]
  
  
  
  kcv_both_aic = lm(price ~ name + distance + surge_multiplier + temperature + hour + windSpeed + apparentTemperature + month + name:distance, data = training_data)
  
  kcv_both_bic  = lm(price ~ name + distance + surge_multiplier + name:distance, data = training_data)
  
  # Obtain RMSE on the 'test' data
  
  resid_both_aic = test_data[,6] - predict(kcv_both_aic, newdata=test_data) 
  RMSE_kcv_both_aic[i] = sqrt(sum(resid_both_aic^2)/nrow(test_data))
  
  resid_both_bic = test_data[,6] - predict(kcv_both_bic, newdata=test_data) 
  RMSE_kcv_both_bic[i] = sqrt(sum(resid_both_bic^2)/nrow(test_data))
}

# Chooses fit_quad 
mean(RMSE_kcv_both_aic)
mean(RMSE_kcv_both_bic)


y_pred_train = predict(lm_both_bic, newdata = data_afterdrop)
mse_train = mean((y_pred_train - data_afterdrop$price)^2)
mse_train

##clean the test data 
data_test = read.csv("Test data.csv")
data_test$price = log(data_test$price)
data_test$name = as.factor(data_test$name)

data_newtest = data_test[-c(which(data_test$name == "Taxi")),]


##################### Prediction ######################

#predict test 
y_pred_test = predict(lm_both_bic, newdata = data_newtest)
mse_test = mean((y_pred_test - data_newtest$price)^2)
mse_test

#plot prediction
plot(y_pred_test,data_newtest$price,pch=10,col="cyan4")
abline(coef=c(0,1),c="orange",lwd=3)
plot(y_pred_train ,data_afterdrop$price,pch=10,col="cyan4")
abline(coef=c(0,1),c="orange",lwd=3)