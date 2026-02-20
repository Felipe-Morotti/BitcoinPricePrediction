import numpy as np      #Numpy arrays are very fast and can performe large computations in a very short time.
import pandas as pd     #Helps load the data frame in a 2D array format and has multiple functions to performe analysis tasks in one go.
#Draw visualizations.
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split #Multiple libraries having pre-implemented functions to perform tasks from data preprocessing to model development and evaluation.
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier #eXtreme Gradient Boosting machine learning algorithm which is one of the algorithms which helps to achieve high accuracy on predictions.
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

#Importing Dataset
df = pd.read_csv('bitcoin.csv')

print(df.head())
print(df.shape) #Based on the result, there are 2713 rows of data and 7 different features or columns.
print(df.describe())
df.info()

#Exploratory Data Analysis is an approach to analyzing the data using visual techniques.
plt.figure(figsize=(15, 5))
#plt.plot(df['Close'])  #Column named "Close", which is std for the closing price of a stock.
plt.title('Bitcoin Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
#plt.show()

#Looking for discrepancies. If the row counts are the same, it means every single day in the dataset had no adjustments.
#If the first number is smaller than the second, it tells exactly how many days did no have adjustments (first - second)
print(df[df['Close'] == df['Adj Close']].shape, df.shape) #Based on the result, all the rows of columns 'Close' and 'Adj Close' have the same data.

#Drop this column because it's redundant.
df = df.drop(['Adj Close'], axis=1)

#Checking for null values.
print(df.isnull().sum()) #Based on the result, there are no null values in the dataset.

#Draw the distribution plot for the continuous features given in the dataset.
features = ['Open', 'High', 'Low', 'Close']
plt.subplots(figsize=(20,10))
#for i, col in enumerate(features):
    #plt.subplot(2, 2, i + 1)
    #sn.distplot(df[col])
#plt.show()

#plt.subplots(figsize=(20,10))
#for i, col in enumerate(features):
    #plt.subplot(2, 2, i + 1)
    #sn.boxplot(df[col], orient='h')
#plt.show()   #There are so many outliers in the data which means that the prices of the stock have varied hugely in a very short period of time.

#Checking with a barplot
#Feature engineering helps to derive some valuable features from the existing ones.

splitted = df['Date'].str.split('-', expand=True)

df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')

#Convert the "Date" column to datetime objects.
df['Date'] = pd.to_datetime(df['Date'])
print(df.head())

data_grouped = df.groupby('year').mean()
#plt.subplots(figsize=(20, 10))
#for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    #plt.subplot(2, 2, i+1)
    #data_grouped[col].plot.bar()
#plt.show()   #Here is possible to see that there are so many outliers in the data because the prices of bitcoin exploded in 2021.

df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
print(df.head())

#Adding some more columns which will help in the training of the model, including a target feature which is a signal whether to buy or not.
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

#Checking whether the target is balanced or not using a pie chart.
#plt.pie(df['target'].value_counts().values,
        #labels=[0, 1], autopct='%1.1f%%')
#plt.show()

#plt.figure(figsize=(10, 10))
#sn.heatmap(df.corr() > 0.9, annot=True, cbar=False)
#plt.show()

#From the above heatmap, there is a high correlation between OHLC, and the added features are not highly correlated
#with each other or previously provided features which means that is good to go and build the model.

features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

#Scaling the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

#Split the data into training and validation (test) sets
X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.3, random_state=42)
#'test size=0.3' means 30% of the data will be used for testing, and 70% for training.

#Model dev
models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]
for i in range(3):
    models[i].fit(X_train, Y_train)

    print(f'{models[i]}:')
    print('Training Accuracy: ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:, 1]))
    print('Validation Accuracy: ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:, 1]))
    print()

ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid, cmap='Blues')
plt.show()