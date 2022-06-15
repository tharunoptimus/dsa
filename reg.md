# Data Science and Analytics

## Library Imports and File Import
```python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

## Remove Warnings 
```python
import warnings
warnings.filterwarnings('ignore')
```

## Import Dataset
```python
data = '/meow.csv'
df = pd.read_csv(data)
```

## Exploratory Analysis
```python
df.head()
df.info()
df.describe()
df.shape
```

## Cleaning
```python
data['high'] = data['high'].str.replace('$', '')
data['high']=pd.to_numeric(data['high'])
data['high'].fillna(value=0,inplace=True)
housing_map = {'yes': 1, 'no': 0}
sampleDF['housing'] = sampleDF['housing'].map(housing_map)
```


## Drop Columns
```python
df.drop(['C_Name'], axis=1, inplace=True)
```

## Find Categorical
```python
categorical = [var for var in df.columns if df[var].dtype=='O']
numerical = [var for var in df.columns if df[var].dtype!='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :', categorical)
df[categorical].head()
# view frequency distribution of categorical variables
for var in categorical:
    print(df[var].value_counts()/np.float(len(df)))
```

## Missing Data
```python
df[categorical].isnull().sum()
```

## Frequency Counts
```python
df[categorical].value_counts()

## Frequency Distribution
for var in categorical: 
    print(df[var].value_counts()/np.float(len(df)))
```

## Finding Cardinality
```python
for var in categorical: 
    print(var, ' contains ', len(df[var].unique()), ' labels')
```

## Date Stuff
```python
df['Date'].dtypes
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
```

## Find Unique
```python
df['Year'].unique()
```

## Find Outliers
```python
IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
print('outliers  < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

## Declare Feature Vector and Target
```python
X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']
```

## Split Data
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# check the shape of X_train and X_test
X_train.shape, X_test.shape
```

## Engineering Outliers
```python
def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])
for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)

# check the Max of a category
X_train.Rainfall.max(), X_test.Rainfall.max()
```

## Encode Categorical
```python
import category_encoders as ce
encoder = ce.BinaryEncoder(cols=['RainToday'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
```

## Feature Scaling
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
```

## Model Training
```python
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)
# fit the model
logreg.fit(X_train, y_train)

logreg.coef_
logreg.score(X_train, y_train)
```

## Predict
```python
y_pred_test = logreg.predict(X_test)

y_pred_test
```

## Accuracy
```python
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
```

## Overfit and Underfit
```python
print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))
```

## Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])
```

## SNS Heatmap
```python
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
sns.heatmap(data.corr(),annot=True)
```

## Classification Stuff
```python
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
classification_error = (FP + FN) / float(TP + TN + FP + FN)
precision = TP / float(TP + FP)
recall = TP / float(TP + FN)
true_positive_rate = TP / float(TP + FN)
false_positive_rate = FP / float(FP + TN)
specificity = TN / (TN + FP)
```


## K Fold
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
# Average Cross Validation Score
print('Average cross-validation score: {:.4f}'.format(scores.mean()))
```

## Hyperparameter Optimization
```python
from sklearn.model_selection import GridSearchCV
parameters = [{'penalty':['l1','l2']}, 
              {'C':[1, 10, 100, 1000]}]
grid_search = GridSearchCV(estimator = logreg,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)
grid_search.fit(X_train, y_train)

# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))
# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))
# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))
print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))
```

## Plots
```python
data['days_to_next_dividend'].plot(kind='density')
sns.distplot(data['percent_change_price'])

```