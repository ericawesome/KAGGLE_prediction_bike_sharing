# 0. Import libraries and data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import seaborn as sns
''' & to surpresse warning messages '''
import warnings
warnings.filterwarnings('ignore')

bike = pd.read_csv('train.csv', index_col=0)
bike.reset_index(inplace=True)
bike['datetime'] = pd.to_datetime(bike['datetime'])

# 1. Train-test-split
X = bike[['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']]
y = bike['count']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape

# 2. Feature-engineering
def feature_engineer(df):
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year

    d = {2011: 0, 2012 : 1}
    df['year'] = df['year'].map(d)

    df_new = df.drop(columns=['temp', 'weather', 'season', 'datetime'])

    return df_new

Xtrain = feature_engineer(Xtrain.copy())
ytrain = np.log(ytrain)
Xtest = feature_engineer(Xtest.copy())
ytest = np.log(ytest)

# 3. Visualization of feature correlation and the bike sales over time
''' Correlation matrix '''
plt.figure(figsize=(12, 8))
sns.heatmap(data=Xtrain.corr().abs(), annot=True, center=0, vmin=0, vmax=1, cmap='Greys')

''' Bike sales over time '''
Xtrain['Count'] = ytrain.copy()
month_hour_demand = pd.DataFrame(Xtrain.groupby(['hour', 'month'])['Count'].sum())
sbn_mh = month_hour_demand.unstack()
plt.figure(figsize=(32, 16))
sns.heatmap(data=sbn_mh, annot=True, xticklabels=range(1, 13), yticklabels=range(1, 25), cmap='autumn', fmt='.2f', linewidths=1)
plt.show()
Xtrain = Xtrain.drop(columns=['Count'])

# 4. Application of GradientBoostingRegressor model
gbm = GradientBoostingRegressor()
gbm.fit(Xtrain, ytrain)
ypred = gbm.predict(Xtest)

# 5. Calculation of the mean-squared-error
score = np.sqrt(mean_squared_log_error(ytest, np.exp(ypred)))

#6. Hyperparameter optimization (GridSearchCV)
my_param_grid = {'n_estimators': range(200, 250, 10), 'max_depth': [4]}
grid = GridSearchCV(gbm, param_grid=my_param_grid)
grid.fit(Xtrain, ytrain)
print(f'\nBest parameters: {grid.best_params_}')
print(f'Score with training data using GradientBoostingRegressor with best estimators: {grid.best_score_} \n-----------------------------------')
ypred_opt = grid.predict(Xtest)

# 7. Calculation of the mean-squared-error
score_opt = np.sqrt(mean_squared_log_error(ytest, np.exp(ypred_opt)))

print(f'The first score applying the model is {score} and the second score after hyperparameter optimization: {score_opt}')
