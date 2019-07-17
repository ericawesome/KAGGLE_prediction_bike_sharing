====================

awesome_prediction
====================

Development of a model for forecasting the demand for bicycles using the Washington DC data set based on the City bike sharing system, which includes hourly rental data for two years.


How to use:

1. Join competition
https://www.kaggle.com/account/login?returnUrl=%2Fc%2Fbike-sharing-demand%2Frules

2. Data import
Importing the data from https://www.kaggle.com/c/bike-sharing-demand/data: 
* train.csv (10886 x 11 features)
* test.csv (6494 x 9 features)
* sampleSubmission.csv (6494 x 2 features - the required format to submit on kaggle: Datetime + predicted amount of bikes)

3. Try the awesome_predictor.py
https://github.com/ericawesome/awesome_prediction/blob/master/awesome_predictor.py

* Take a close look at the correlation of the given features:
![Correlation_matrix_features](https://user-images.githubusercontent.com/48921737/61363737-d1c4dd00-a884-11e9-9a6d-710e409fc142.png)

* Take a look at the distribution of bicycle needs:
![Heatmap_bike_demand_per_hour_all_year](https://user-images.githubusercontent.com/48921737/61363739-d1c4dd00-a884-11e9-8f3c-a520c9245d81.png)

* What are you willing to do next?


4. Finally adjust the features and try to run different ML-models and see your scoring on kaggle.
https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
