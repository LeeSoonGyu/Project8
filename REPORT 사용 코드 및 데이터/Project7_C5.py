## Project7
# Numpy/Pandas 와 R 비교

# 환경설정
all = [var for var in globals() if var[0] != "_"]
for var in all:
    del globals()[var]

import numpy as np
import pandas as pd

# 공통 데이터
data1 = [1.5, 4, 2, 3]
data1
data2 = [[1, 2, 3, 4],[5, 6, 7, 8]]
data2

# 1. empty(numpy)
data1 = np.empty((2, 2)) # 2*2 배열함수 생성 함수
data1
np.empty((2, 4, 5))

# empty(pandas)
df_empty = pd.DataFrame({'A' : []}) # 인덱스가 비어있는것을 확인하는 함수
df_empty
df_empty.empty

df = pd.DataFrame({'A' : [np.nan]}) # 인덱스에 nan만 있으면 비어있는것으로 간주 하지 않는다.
df
df.empty
df.dropna().empty # NA를 삭제하면 인덱스가 비워진 상태로 출력

# 2. sum(numpy)
np.sum(data1) # 데이터 안에 들어있는 모든 원소의 합 추출 함수
np.sum(data2)
# sum(pandas)
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
              [np.nan, np.nan], [0.75, -1.3]],
                index=['a', 'b', 'c', 'd'],
                columns=['one', 'two'])
df

df.sum()
df.sum('index')
df.sum('columns') # nan = 0으로 자동변환

# 3. mean(numpy)
np.mean(data1) # 데이터 안에 있는 모든 원소의 평균
np.mean(data2)
# mean(pandas)
df
df.mean()
df.mean('index')
df.mean('columns')

# 4. zeros(numpy)
np.zeros(10) # 주어진 길이나 모양에 0이라는 값으로 배열 생성
np.zeros((3, 6))
np.zeros((2, 3, 2))
# zeros(pandas)
a = pd.DataFrame(np.zeros(10))
b = pd.DataFrame(np.zeros((3, 6)))
c = pd.DataFrame(np.zeros(2,3,2)) # 3차원 부터는 출력의 지원이 안된다.
print(a)
print(b)

# 5. var(numpy)
np.var(data1, ddof=1) # 데이터 안에 있는 원소들의 분산 값 / ddof=1 (표준편차를 계산할 때, n-1로 나누라는 의미)
np.var(data2, ddof=1) # https://www.abbreviationfinder.org/ko/acronyms/ddof.html#aim
# var(pandas)
df.var()
df.var('index')
df.var('columns')

# 6. min(numpy)
np.min(data1)
np.min(data2)
# min(pandas)
df.min()
df.min('index')
df.min('columns')

# 7. max(numpy)
np.max(data1)
np.max(data2)
# max(pandas)
df.max()
df.max('index')
df.max('columns')

# 8. cumsum(numpy)
np.cumsum(data1) # 데이터 안에 있는 원소들의 누적 합의 추출 함수 (sum과는 별개)
np.cumsum(data2)
# cumsum(pandas)
df
df.cumsum('index')
df.cumsum('columns')

# 9. random(numpy)
data = np.random.randn(2, 3) # 난수 생성
data

# random(pandas)
df1 = pd.DataFrame(np.random.randn(2, 3))
df1

# 10. sqrt(numpy)
np.sqrt(data1)
np.sqrt(data2)
# sqrt(pandas)
df.dropna(axis=0)
df.dropna(axis=1)
df.transform('sqrt')



# 11. sort(numpy)
data1
sortData1 = np.sort(data1)
sortData1

# 12. append(numpy)
## numpy append는 내장 append와는 다르게 차원의 수가 맞지 않으면 이어붙일 수 없다.
x=np.array(data1)
y=np.array(data2)

print(x.shape)
print(y.shape)

append1 = np.append(x, y.reshape(1, 2), axis=0)
append2 = np.append(y, x.reshape(1, 4), axis=0)
append1
append2



# 13. delete(numpy)
# 형식) np.delete(arr, obj, axis=None)
# The axis along which to delete the subarray defined by obj.
# If axis is None, obj is applied to the flattened array.
arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
arr
np.delete(arr, 1, 0)
np.delete(arr, np.s_[::2], 1)
np.delete(arr, [1,3,5], None)



# 14. copy(numpy)
## 형식)np.copy(data, order='K', subok=False)
x = data1
y = np.copy(data1)

x == data1
y == data1

## np.copy는 얕은복사이기 때문에 x의 내용이 data2로 바뀌어도
 # 기존에 복사했던 data1의 데이터를 그대로 갖고 있는다.
x = data2
x == data1
y == data1




# 15. arange(numpy)
## 형식)numpy.arange([start, ]stop, [step, ]dtype=None, *, like=None)
### 내장 함수인 range와 동일한 기능이지만, 결과는 list가 아닌 ndarray로 생성된다.
np.arange(3)
np.arange(3.0)
np.arange(3,7)
np.arange(3,7,2)




# 16. read_csv(pandas)
pd.read_csv('C:/Users/junseo/OneDrive/code/Project/Project7/testdata/ev_stations_v1.csv')




# 17. unique(Numpy)(axis -> 변경)
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names) #dtype을 따로 표기해주는 이유: 숫자형이 아니기 때문으로 예상됨
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)



# 18. dtype(numpy)
# ndarray는 동종 데이터를 위한 일반 다차원 컨테이너이다.
# 즉, 모든 요소는 동일한 유형이어야 한다.
# 모든 배열은 모양, 각 차원의 크기를 나타내는 튜플 및 배열의 데이터 유형을 설명하는 객체인 dtype을 가진다.
npdata1 = np.array(data1)
npdata1.dtype
npdata2 = np.array(data2)
npdata2.dtype




# 19. slicing
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[1:])

## pandas
df_empty = pd.DataFrame({'A' : []})
df_empty
df_empty.empty

### cf)NaN이 들어가 있을 경우 비어 있지 않은 것으로 간주한다.
dfNaN = pd.DataFrame({'A' : [np.nan]})
dfNaN
dfNaN.empty




# 20. dataframe(pandas)
# 형식)pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)
## parameters
### data: ndarray, dict, dataframe
    # => data가 dict개체라면 열순서는 입력순서를 따른다.
### index: 결과에 사용할 인덱스. none으로 설정하면 데이터의 인덱싱 정보 부분 및 인덱스가 제공되지 않음
### columns: 데이터에 열 레이블이 없는 경우 결과 프레임에 사용할 열 레이블,
            # 범위 인덱스로 기본값 설정(0,1,2,...,n) 데이터에 열 레이블이 포함된 경우 대신 열 선택을 수행한다.
### dtype: 강제 적용할 데이터 유형. 단 하나의 타입만 허용된다.
### copy: true or false로 설정 가능하며, None의 기본값은 false이다.
pd.DataFrame(data1,index=(1,2,3,4), columns=None, dtype=None, copy=None)
pd.DataFrame(data2,index=(1,2), columns=None, dtype=None, copy=None)



# 공통 데이터
data = {'a' : [1,2,3], 'b' : [4,5,6]}
c = pd.Series([1,2,pd.NA,4], index=(1,2,3,4))
print(c)
print(data)


df = pd.DataFrame(data, index=(1,2,3))
print(df)

#21. ndim (pandas)
print(df.ndim)

#22. size (pandas)
print(df.size)

#23. values (pandas)
print(df.values)

#24. head (pandas)
print(df.head())

#25. tail (pandas)
print(df.tail())

#26. shape (pandas)
print(df.shape)

#27. T(Transpose) (pandas)
print(df.T)

#28. describe (pandas)
print(df.describe())

#29. notnull (pandas)
print(c[c.notnull()])

#30. get_dummies()
pd.get_dummies(df['a'])