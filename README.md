# 빅데이터 분석기사
빅데이터 분석기사를 준비하면서 공부한 내용 정리입니다

## 유형 2 분류 분석

문제를 읽고 종속변수(타겟변수)의 예측이 범주형 변수인 경우 분류 분석을 진행한다.

* 주의할 점은 nunique(), describe(include='object')를 사용해 고유 값의 수를 확인하여 원-핫 인코딩을 진행하고
* 고유 값이 너무 많을 경우 적절한 drop 처리나 그래도 많을 경우 LabelEncoder를 진행해야 한다.

 
```python
train = pd.read_csv("",encoding = 'euc-kr')
# EDA 진행

df.shape # 행열 확인
df.isnull().sum # 결측값 확인
df.descibe(), df.describe(include='object') # 기초통계
df.nunique() # 각 열의 개수확인
y['gender'].value_counts() # 종속변수 값 확인

# 데이터 전처리

y = y['gender] # 종속변수 선언
x = train.drop(columns = ['회원ID','gender']) # 학습 데이터 설정
test_x = test.drop(columns = ['회원ID']) # 평가 데이터 설정

# 데이터 더미화

x_dum = pd.get_dummies(x) # 학습 데이터 독립변수 더미화
test_dum = pd.get_dummies(test_x).reindex(columns = x_dum.columns, fill_value = 0) # 평가 데이터 더미화

# 데이터 분할

from sklearn.model_selection import train_test_split
# 학습용 평가용 데이터 분할, 평가 데이터 20%, 학습 데이터 80% 할당
x_train, x_test, y_train, y_test = train_test_split(x_dum, y, test_size = 0.2, random_state = 10)

# 학습 데이터 모델링

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# 랜덤포레스트 기본 세팅 및 학습
rf = RandomForestClassifier(random_state = 10)
rf.fit(x_train, y_train)

# 모델링 예측 및 평가

# 만약 타겟 데이터에 대한 예측이 확률 예측이라면 predict_proba()를 사용한다

pred = rf.predict(x_test)
print(f1_score(y_test, average = 'macro')
pred_test = rf.predict(test_dum)

# 결과를 출력받고 이를 프레임화 시키고 저장한다
# 데이터 프레임화 및 저장

# 데이터 프레임화 및 csv 파일 저장
output = pd.DataFrame({'ID' : test['ID'], 'gender' : pred_test})
output.to_csv("성별예측.csv", index = False)
```

## 유형 2 회귀 분석

* 만약 문제에서 구하고자하는 타겟 변수의 값이 연속형이라면 회귀분석을 진행한다.
* 보통 판매 가격 예측과 같은 연속형 변수를 의미한다

```python
import pandas as pd
train = pd.read_csv('.csv')
test = pd.read_csv('.csv')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
# 전처리
y = train['price']
x = train.drop(columns = ['ID', 'price'])
test_x = test.drop(columns = ['ID'])
# 더미
x_dum = pd.get_dummies(x)
test_dum = pd.get_dummies(test_x).reindex(columns = x_dum.columns, fill_value = 0)
# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x_dum, y, test_size = 0.2,random_state = 5)
# 학습 모델
rr = RandomForestRegressor(random_state = 5)
rr.fit(x_train, y_train)
# 예측 및 평가
pred = rr.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred)))
pred_test = rr.predict(test_dum)
# 데이터프레임화 csv 저장
output = pd.DataFrame({'ID':test['ID'], 'price':pred_test})
output.to_csv("회귀분석.csv", index = False)
```

## 유형 3 가설 검정(정규성, 등분산, 표본검정)

귀무가설(영가설)은 기존에 정립돼 있는 당연한 이치이고, 대립가설은 기존 이치에 벗어나는 가설이다. 

### 1. 가설 설정
### 2. 정규성 검정
정규성 검정에는 크게 3가지 검정 방법이 있다.
* Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov 정규성 검정

표본의 크기에 따라 5000개 미만(샤피로), 5000개 이상(앤더슨)으로 나뉜다. 

```python
from scipy.stats import shapiro
from scipy.stats import anderson
from scipy.stats import kstest

stat, p_value = shapiro(data)
result = anderson(data)
stat, p_value = kstest(data, 'norm')
```
코드는 다음과 같고, 샤피로는 검정 통계량 p값 > 0.05, 귀무가설 채택하지만 앤더슨의 경우 c값(critical_values) > 0.05 일 경우 귀무가설을 채택한다.

### 3. 등분산 검정

등분산은 분산이 일정한지에 대한 검정으로 정규성 검정 후, p값에 따라 정규성을 따르는 T 검정(독립표본 t-검정,대응표본 t-검정)을 선택하거나, 정규성을 가지지 않을 경우 비모수 검정응로 윌콕슨 순위 부호, 맨휘트니 부호 검정을 실시한다.

* levene, bartlett 등분산 검정

```python
from scipy.stats import levene
from scipy.stats import bartlett

# 두 독립표본 데이터를 가진 변수 A와 B
statistic, p_value = levene(A, B)
statistic, p_value = bartlett(A, B)

```

### 등분산 유무에 따른 검정

* 정규성을 가지는 t검정
```python
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
# 두 대응표본 데이터를 가진 변수 A와 B
statistic, p_value = ttest_rel(A, B)
# 두 대응표본 데이터를 가진 변수 A와 B
statistic, p_value = ttest_ind(A, B)
```
```python
from scipy.stats import ttest_ind

# 두 독립표본 데이터를 가진 변수 A와 B
statistic, p_value = ttest_ind(A, B, equal_var=False)
```

* 정규성을 가지지 않을 경우 비모수 검정

```python
from scipy.stats import mannwhitneyu

# 두 독립표본 데이터를 가진 변수 A와 B
statistic, p_value = mannwhitneyu(A, B)
```


### 범주형 변수 간의 관련성 검정

두 변수 간의 관련성이 있다고 판단할 수 있는 근거를 찾는 코드이다.


```python
from scipy.stats import chi2_contingency

# 분할표 (contingency table) 데이터
observed_data = [[observed_frequency11, observed_frequency12],
                 [observed_frequency21, observed_frequency22]]

# 카이제곱 검정 수행
statistic, p_value, _, _ = chi2_contingency(observed_data)

# observed_data: 분할표 (contingency table) 데이터. 각 셀은 각 범주의 빈도를 나타낸다
```


