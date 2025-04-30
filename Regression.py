import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

bike_df = pd.read_csv("Bike-day.csv")
bike_df.head()
bike_df.shape
bike_df.info()
bike_df.describe()
percent_missing = bike_df.isnull().sum() * 100 / len(bike_df)
print(percent_missing)
bike_df.duplicated()
def check_value_counts(bike_df):
    for column in bike_df.columns:
        print(bike_df[column].value_counts())
bike1_df = bike_df.drop('instant',axis=1)
check_value_counts(bike1_df)
bikenew_df = bike_df.drop(['instant','dteday', 'casual', 'registered'], axis=1)
bikenew_df.head()
selected_vars = ['cnt', 'temp', 'atemp','hum', 'windspeed']
sns.pairplot(bikenew_df, vars=selected_vars)
plt.show()
sns.boxplot(x ='season', y ='cnt', data = bikenew_df, palette = 'Set1')
plt.show()
sns.boxplot(x ='yr', y ='cnt', data = bikenew_df, palette = 'Set1')
plt.show()
sns.boxplot(x ='holiday', y ='cnt', data = bikenew_df, palette = 'Set1')
plt.show()
sns.boxplot(x ='weekday', y ='cnt', data = bikenew_df, palette = 'Set1')
plt.show()
sns.boxplot(x ='workingday', y ='cnt', data = bikenew_df, palette = 'Set1')
plt.show()
sns.boxplot(x ='weathersit', y ='cnt', data = bikenew_df, palette = 'Set1')
plt.show()
sns.boxplot(x ='mnth', y ='cnt', data = bikenew_df, palette = 'Set1')
plt.show()
import calendar

bike_new['mnth'] = bike_new['mnth'].apply(lambda x: calendar.month_abbr[x])
bike_new.season = bike_new.season.map({1: 'Spring',2:'Summer',3:'Fall',4:'Winter'})
bike_new.weathersit = bike_new.weathersit.map({1:'Clear',2:'Mist & Cloudy', 
                                             3:'Light Snow & Rain',4:'Heavy Snow & Rain'})
bike_new.weekday = bike_new.weekday.map({0:"Sunday",1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thrusday",5:"Friday",6:"Saturday"})
bike_new.head()
import calendar

bikenew_df['mnth1'] = bikenew_df['mnth'].apply(lambda x: calendar.month_abbr[x])
bikenew_df['season1'] = bikenew_df['season'].map({1: 'Spring',2:'Summer',3:'Fall',4:'Winter'})
bikenew_df['weathersit1'] = bikenew_df['weathersit'].map({1:'Clear',2:'Mist & Cloudy', 
                                             3:'Light Snow & Rain',4:'Heavy Snow & Rain'})
bikenew_df['weekday1'] = bikenew_df['weekday'].map({0:"Sunday",1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",5:"Friday",6:"Saturday"})
bikenew_df.head()
print(bikenew_df)
bikenew_df = bikenew_df.drop(['season','mnth', 'weathersit', 'weekday'], axis=1)
bikenew_df.head()
from sklearn.model_selection import train_test_split
biketrain_df, biketest_df = train_test_split(bikenew_df, test_size = 0.2)
print(biketrain_df)
print(biketest_df)
from sklearn.preprocessing import StandardScaler
columns_to_standardize = ['cnt', 'temp','atemp', 'hum','windspeed','yr','holiday','workingday']
scaler = StandardScaler()
biketrain_df[columns_to_standardize] = scaler.fit_transform(biketrain_df[columns_to_standardize])
print(biketrain_df)
# Apply scaler() to all the columns except the 'dummy' variables.
tempbikenew_df = biketrain_df.drop(['season1','mnth1', 'weathersit1', 'weekday1'], axis=1)
sns.scatterplot(data = biketrain_df, x = 'temp', y = 'cnt')
sns.scatterplot(data = biketrain_df, x = 'atemp', y = 'cnt')
tempbikenew_df.corr()
features = biketrain_df.columns.tolist()
features.remove("cnt")
print(features)
X = biketrain_df[features]
y = biketrain_df["cnt"]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
model = LinearRegression()
rfe = RFE(estimator=model, n_features_to_select=15)
rfe.fit(X,y)
selected_features = rfe.support_
ranking = rfe.ranking_

print("Selected Features: ", selected_features)
print("Feature Ranking: ", ranking)

X_train = X[['yr', 'temp', 'mnthJan', 'mnthFeb','mnthJul', 'mnthSep',  'mnthNov', 'mnthDec', 'seasonSpring', 'seasonWinter', 'weathersitClear', 'weathersitMist & Cloudy', 'weathersitLight Snow & Rain',  'weekdaySunday', 'weekdayMonday']]
print(X_train)
y_train = y
model = LinearRegression().fit(X_train, y_train)
print(f"intercept:{model.intercept_}")
print(f"coefficients:{model.coef_}")
r_squared = model.score(X_train,y_train)
print(f"R_squared:{r_squared}")
vif = 0
if r_squared < 1:
  vif = 1 / (1 - r_squared)
print(vif)
for column in X_train.columns.tolist():
  model1 = LinearRegression().fit(X_train[[column]], y_train)
  r_squared = model1.score(X_train[[column]],y_train)
  print(f"R_squared:{r_squared}")
  vif = 0
  if r_squared < 1:
    vif = 1 / (1 - r_squared)
  print(vif)
  lm_2 = LinearRegression().fit(X_train, y_train)
  r_squared = lm_2.score(X_train,y_train)
print(f"R_squared:{r_squared}")
vif = 1 / (1 - r_squared)
print(vif)
y_pred = lm_2.predict(X_train)
y_pred
residuals = y_train - y_pred
sns.histplot(residuals, kde=True)
import scipy.stats as stats
qq_data = stats.probplot(residuals, dist="norm")
plt.figure(figsize=(8, 6))
plt.plot(qq_data[0][0], qq_data[0][1], 'o', label='Sample Data')
plt.plot(qq_data[0][0], qq_data[0][0], 'r--', label='Theoretical Line')
plt.title('Q-Q Plot of Error Terms')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.legend()
plt.grid(True)
plt.show()
from sklearn.preprocessing import StandardScaler
columns_to_standardize = ['cnt', 'temp','atemp', 'hum','windspeed','yr','holiday','workingday']
biketest_df[columns_to_standardize] = scaler.transform(biketest_df[columns_to_standardize])
print(biketest_df)
features = biketest_df.columns.tolist()
features.remove("cnt")
X_test = biketest_df[features]
y_test = biketest_df["cnt"]
X_test = X_test[['yr', 'temp', 'mnthJan', 'mnthFeb','mnthJul', 'mnthSep',  'mnthNov', 'mnthDec', 'seasonSpring', 'seasonWinter', 'weathersitClear', 'weathersitMist & Cloudy', 'weathersitLight Snow & Rain',  'weekdaySunday', 'weekdayMonday']]
X_test.head()
y_pred = lm_2.predict(X_test)
y_pred
residuals = y_test - y_pred
r_squared = lm_2.score(X_test,y_test)
print(f"R_squared:{r_squared}")
plt.scatter(y_test,y_pred)

