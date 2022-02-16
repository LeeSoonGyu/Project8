# Project7 OJS ####
## 기간: 22.01.18(화) 9:30까지
# C5조: 이순규 오준서 임성현

# 환경설정
rm(list=ls())
getwd()
setwd('c:/rwork')

# 라이브러리 모음
install.packages("data.table") #copy()

library(data.table)


# 공통 데이터
data1 <- array(c(1.5, 4, 2, 3))
data2 <- array(c(1:8),c(1,4,2))


# 비교 데이터 생성

# numpy 비교 데이터
data1 <- c(1.5, 4, 2, 3) 
data1

vec1 <- c(1:8)
data2 <- array(vec1, c(1, 4, 2))
data2
data2[1,1:4,1]

# pandas dataframe 비교 데이터
own <- c(1.4, 7.1, NA, 0.75)
two <- c(NA, -4.5, NA, -1.3)
df <- data.frame(own, two)
df
rownames(df) <- c('a', 'b', 'c', 'd')
df

# empty
install.packages('plyr') # empty함수를 쓰기 위한 패키지 설치
library(plyr)

# pandas 비교
empty(df)

# numpy 비교
empty(data1)
empty(data2) # 데이터 프레임이 비었는지 확인 하는 함수
# pandas의 empty의 함수와 동일하게 쓰인다.

# sum
# numpy 비교
sum(data1) # 데이터 안에 있는 모든 원소의 합
sum(data2)

# pandas 비교
sum(df$own, na.rm = T)
sum(df$two, na.rm = T)
rowSums(df, na.rm = T)
# numpy, pandas와 동일하게 쓰인다.

# mean
# numpy 비교
mean(data1) # 데이터 안에 있는 모든 원소의 평균
mean(data2)

# pandas 비교
mean(df$own, na.rm = T)
mean(df$two, na.rm = T)
rowMeans(df, na.rm = T)
# numpy, pandas와 동일하게 쓰인다.

# zeros
install.packages('phonTools')
library(phonTools)

zeros(10)
zeros(3, 6)
zeros(2, 3, 2) # 3차원 지원 출력이 안됨

# var
# numpy 비교
var(data1)
var(data2)

# pandas 비교
var(df$own, na.rm = T)
var(df$two, na.rm = T)
apply(df, 1, var, na.rm = TRUE)

# min
# numpy 비교
min(data1) # 데이터 안에 있는 원소중 제일 작은 값
min(data2)

# pandas 비교
min(df$own, na.rm = T)
min(df$two, na.rm = T)
apply(df, 1, min, na.rm = TRUE)

# max
# numpy 비교
max(data1) # 데이터 안에 있는 원소중 제일 큰 값
max(data2)

# pandas 비교
max(df$own, na.rm = T)
max(df$two, na.rm = T)
apply(df, 1, max, na.rm = TRUE)

# cumsum
# numpy 비교
cumsum(data1) # 데이터 안에 있는 원소들의 누적 합
cumsum(data2)

# pandas 비교 na값을 전처리 해야함 그래야 비교가능
is.na(df)
df1 <- na.omit(df)
cumsum(df1$own)
cumsum(df1$two)
apply(df1, 1, cumsum)
# numpy, pandas와 동일하게 쓰인다.

# random
rnorm(n=3, mean=0, sd=2)

# sqrt
# numpy 비교
sqrt(data1)
sqrt(data2)

# pandas 비교
sqrt(df$own)
sqrt(df$two)
apply(df, 1, sqrt)





# 11. sort(numpy)
data1
x11 <- sort(data1)
x11




# 12. append(numpy)
x12 <- data1
x12
x12 <- append(x12,data2)
x12




# 13. delete(numpy)
data2
data2[,,-1]
(data2[,-c(2,4),]) #? 차원이 추가되면 전치행렬로 결과를 출력함
data2[!data2 %in% c(1,3,5)] #? 차원 유지 가능?


# 14. copy(numpy)
x = data1
y = copy(x)

x == data1
y == data1


x = data2
x == data1 # 변수내용이 data2로 바뀌자 오류발생
y == data1 # 얕은 복사를 했기때문에 x의 내용변경과는 무관




# 15. arange(numpy)
# r에서 range함수는 단지 최대 최소값만을 나타낸다.
range(1,3,6)
range(1.0:3.0,0.1)

# 파이썬과 같이 정해진 규칙으로 배열을 생성하려면 seq를 이용한다.
seq(1,3,by=1)
seq(1.0,3.0,by=1.0) # 소수단위는 생성하지 않는다.
seq(3.0,6.5) # by옵션 미설정시 기본값은 1이다.
seq(3,6,2)




# 16. read_csv(pandas)
read.csv('C:/Users/junseo/OneDrive/code/Project/Project7/testdata/ev_stations_v1.csv')




# 17. unique(numpy)
names = c('Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe')
unique(names)
ints = c(3, 3, 3, 2, 2, 1, 1, 4, 4)
unique(ints)




# 18. dtype(numpy)
mode(data1)
mode(data2)
typeof(data1)
typeof(data2)




# 19. slicing
arr <- array(c(1, 2, 3, 4, 5, 6, 7))
print(arr[1:5])




# 20. dataframe(pandas)
dataframe1 <- as.data.frame(data1)
dataframe1
dataframe2 <- as.data.frame(data2)
dataframe2
## 파이썬과는 다르게 2차원의 배열을 1차원으로 묶어서 데이터 프레임을 형성하였다.



# 임성현

# 공통데이터
a = c(1,2,3)
b = c(4,5,6)
df = data.frame(a, b)
df
c = c(1,2,NA,4)

#21 ndim ; 차원만 따로 보여주는 기능을 못찾아서 dim으로 대체
dim(df)

#22 size
nrow(df)*ncol(df)

#23 values
array(c(a,b), dim = c(3,2,1))

#24 head
head(df)

#25 tail
tail(df)

#26 shape
dim(df)

#27 T
t(df)

#28 describe
summary(df)

#29 notnull
na.omit(c)

#30 get_dummies
table(df$a)


