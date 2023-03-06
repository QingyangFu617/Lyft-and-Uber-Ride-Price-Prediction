# Lyft-and-Uber-Ride-Price-Prediction
This is a multiple linear regression analysis project using R. The purpose of our study is to predict the price of each ride based on given conditions such as time, type of vehicle selected, and weather indexes. 

In this project, the dataset we have selected contains more than 10000 records of rides. Then, we decided to reduce our dataset into a randomly selected subset with 4042 observations. Within the dataset, we have one target variable, price, and 58 predictors.

We choose to construct multivariate linear models for this dataset due to the model’s ability to explain the relationship between one response variable and many predictor variables. As figure 1 shows that distance has a linear relationship with the target variable price.
![1 intro : linear relationship](https://user-images.githubusercontent.com/100692852/223024192-c3ec14ea-2fbb-471c-97c3-005f0e6927f6.png)

This study aims to construct the optimal multivariate linear regression model to predict each ride’s price, and we build our models from different selection algorithms. Forward, backward, and bidirectional selection algorithms will be utilized to generate several models for comparisons. Within the selection algorithms, we also utilized various measurements to select models, such as AIC and BIC. After building the models, we choose our best and final model based on the lowest RMSE from cross-validation and the highest score of adjusted R 2 to avoid multicollinearity. 

<img width="367" alt="截屏2023-03-06 00 21 56" src="https://user-images.githubusercontent.com/100692852/223025480-4526eae7-3a5b-449f-91e9-fa0cbdf1acad.png">


After comparison, our best and final model is the Fit Both BIC. This model contains 24 predictors, name, distance, and surge multiplier, along with the interaction term between name and distance.

price ∼ name + distance + surge multiplier + name × distance

<img width="554" alt="截屏2023-03-06 00 23 12" src="https://user-images.githubusercontent.com/100692852/223025675-9d8ac889-f1fb-4a52-822e-6ea82f992097.png">

Name indicates which type of vehicle the user selected. Luxurious or larger-sized vehicles have a higher pricing strategy than shared rides and regular vehicle models. The coefficients of predictor UberPool and UberXL is -0.767 and -0.277, meaning that UberPool has a greater effect on price than UberXL. Considering the coefficients are negative, when other predictors 8 are fixed, the price with category UberPool will be lower than that with UberXL. In reality, the price of UberPool ride is actually cheaper than UberXL assuming other factors remain the same. distance is the most self-explanatory attribute; the longer the trip, the higher the fare. The positive 0.1725 coefficient of distance precisely reflects this relationship. However, the level of impact of distance is lower than excepted. The potential reason is that the the effect of distance is relatively stable and consistent, and the type of vehicle has more significant impact on price. Surge multiplier indicates a multiplier of the price of each ride based on the increasing demand for rides; the higher the demand, the higher the multiplier. Lastly, the interaction terms between name and distance are also significant, as different types of vehicles may have different pricing strategies per unit distance traveled; hence, such interaction can heavily influence the prediction of the target variable, price. Overall, this model and the coefficients of predictors are performing in the logical sense.
