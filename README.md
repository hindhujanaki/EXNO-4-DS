# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
       
## Feature Scaling
```
import pandas as pd
from scipy import stats
import numpy as np
```
```
import pandas as pd
df=pd.read_csv("/content/bmi.csv")
df
```

```
df.head()
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/d021050b-e88c-405a-93d7-2eda73b806ef)

```
import numpy as np
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/958f99b3-cf31-412f-9dc1-9040c4b4a3e7)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/89c7d9b2-0a03-4b8c-8551-4edec1f07e7d)

```
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/018758cc-322b-4ba3-812a-a2830813c89b)

```
from sklearn.preprocessing import Normalizer
Scaler=Normalizer
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/548f24d7-0fec-4e92-bb40-f678cb55c111)

```
df=pd.read_csv("/content/bmi.csv")
```
```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/c62d9316-5b94-4de0-828f-8c1204a21b0f)


## Feature Selection
```
import pandas as pd
import numpy as np
import seaborn as sns
```
```
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
```
```
data = pd.read_csv("/content/income(1) (1).csv")
data
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/af169232-e1b2-46d5-90a4-e7361b1fc5c5)

```
data.isnull().sum()
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/41ff0cb0-95cc-4f9d-8bd2-ee9cbcff961a)

```
missing = data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/f28637fa-027f-48b3-b239-7be6132ac813)

```
data2=data.dropna(axis=0)
data2
```

```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/2cb5b78d-77d8-456a-a7a6-5426f8c14ee0)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/0733c032-e577-421c-aaea-bb0207d76571)

```
data2
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/46ce2eac-c2b0-4442-8fab-1cf438d00a55)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/cdf4f75b-39ac-4dd0-ae30-144388735625)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/35ce474c-19c2-4e61-9872-c5b49171208c)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/249acca0-5e5f-440c-9b31-b02a29252d93)


```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/98914033-a595-44e6-9e41-667b185e9289)

```
x = new_data[features].values
print(x)
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/42728654-397c-4cf0-a746-21f827910583)

```
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state = 0)
KNN_classifier = KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x, train_y)
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/54bfffac-4de7-4e57-b356-a53d9fb97a9d)

```
prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/397454ec-0127-4364-a621-60dceb19c545)

```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/73f7c5f4-1b2b-4a77-80b9-8f719ea76a2a)

```
print( 'Misclassified samples: %d' % (test_y != prediction).sum())
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/51b7a3bb-bbf4-475c-a5da-10ca6908c761)

```
data.shape
```
![image](https://github.com/hindhujanaki/EXNO-4-DS/assets/148514666/4ec07f96-841c-49fb-aee9-b3710a473612)



# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.

