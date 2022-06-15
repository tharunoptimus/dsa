# Support Vector Machine

## Import Stuff
```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
%matplotlib inline
```

## Import CSV
```python
data = '/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv'
df = pd.read_csv(data)
```

## Draw Histogram to check data distribution
```python
plt.figure(figsize=(24,20))
plt.subplot(4, 2, 1)
fig = df['IP Mean'].hist(bins=20)
fig.set_xlabel('IP Mean')
fig.set_ylabel('Number of pulsar stars')
```

## Declare Feature Vector
```python
X = df.drop(['target_class'], axis=1)
y = df['target_class']
```

## Split Data
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

## Feature Scaling
```python
cols = X_train.columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
```

## Run SVM
```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# instantiate classifier with default hyperparameters
svc=SVC() 
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
```

## SVM with options
```python
svc=SVC(C=100.0) # RFB Kernel with 100C ⬆C == ⬇Outliers
svc=SVC(C=1000.0) 
linear_svc=SVC(kernel='linear', C=1.0) 
linear_svc100=SVC(kernel='linear', C=100.0) 
linear_svc1000=SVC(kernel='linear', C=1000.0) 
poly_svc=SVC(kernel='poly', C=1.0)
sig_svc=SVC(kernel='sigmoid', C=100.0)
```

## Class Distribution
```python
y_test.value_counts()
```

