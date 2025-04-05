---
title: "Python review basic"
excerpt: "From Hello world to basic library"

categories: # 카테고리 설정
  - categories1
tags: # 포스트 태그
  - [tag1, tag2]

permalink: /categories1/Python review basic/ # 포스트 URL

toc: true # 우측에 본문 목차 네비게이션 생성
toc_sticky: true # 본문 목차 네비게이션 고정 여부

date: 2025-03-27 # 작성 날짜
last_modified_at: 2025-03-27 # 최종 수정 날짜
---

# Python review basic from Hello world to basic library


---

# 1. 기본 문법

## 변수 및 자료형
```python
# 변수 선언
x = 5         # 정수  
y = 3.14      # 실수  
name = "John" # 문자열  
is_active = True # 불린값
```

## 자료형 변환
```python
int("5")       # 5 (문자열 -> 정수)  
str(5)         # "5" (정수 -> 문자열)  
float("3.14")  # 3.14 (문자열 -> 실수)
```

## 리스트, 튜플, 딕셔너리
```python
# 리스트
numbers = [1, 2, 3, 4]

# 튜플 (불변)
coordinates = (10, 20)

# 딕셔너리
person = {"name": "Alice", "age": 25}
```

## 조건문
```python
if x > 10:
    print("x는 10보다 큽니다.")
elif x == 10:
    print("x는 10입니다.")
else:
    print("x는 10보다 작습니다.")
```

## 반복문
```python
# for 문
for i in range(5):
    print(i)

# while 문
count = 0
while count < 5:
    print(count)
    count += 1
```

## 함수 정의
```python
def greet(name):
    return f"Hello, {name}!"

result = greet("John")
print(result)
```

---

# 2. 객체지향 프로그래밍

## 클래스와 객체
```python
class Car:
    def __init__(self, model, year):
        self.model = model
        self.year = year

    def start(self):
        print(f"{self.model} is starting!")

my_car = Car("Toyota", 2020)
my_car.start()
```

## 상속
```python
class Animal:
    def speak(self):
        print("Animal sound")

class Dog(Animal):
    def speak(self):
        print("Bark")

dog = Dog()
dog.speak()  # "Bark"
```

---

# 3. 고급 기능

## 제너레이터
```python
def my_generator():
    yield 1
    yield 2
    yield 3

gen = my_generator()
for value in gen:
    print(value)
```

## 데코레이터
```python
def my_decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

## 람다 함수
```python
# 일반 함수
def add(x, y):
    return x + y

# 람다 함수
add_lambda = lambda x, y: x + y
print(add_lambda(5, 10))  # 15
```

## 리스트 컴프리헨션
```python
squares = [x**2 for x in range(10)]
print(squares)
```

---

# 4. 자료구조

## 스택
```python
stack = []
stack.append(1)  # push
stack.append(2)
print(stack.pop())  # pop -> 2
```

## 큐
```python
from collections import deque
queue = deque()
queue.append(1)  # enqueue
queue.append(2)
print(queue.popleft())  # dequeue -> 1
```

## 해시맵
```python
hash_map = {}
hash_map["key"] = "value"
print(hash_map["key"])  # "value"
```

---

# 5. 데이터베이스 연결 (PostgreSQL 예시)
```python
import psycopg2

# 데이터베이스 연결
conn = psycopg2.connect(
    dbname="your_db", user="your_user", password="your_password", host="localhost", port="5432"
)

# 커서 생성
cur = conn.cursor()

# 쿼리 실행
cur.execute("SELECT * FROM your_table")
rows = cur.fetchall()

for row in rows:
    print(row)

# 연결 종료
cur.close()
conn.close()
```

---

# 6. 파일 입출력
```python
# 파일 쓰기
with open('file.txt', 'w') as file:
    file.write("Hello, world!")

# 파일 읽기
with open('file.txt', 'r') as file:
    content = file.read()
    print(content)
```

---

# 7. 예외 처리
```python
try:
    x = 5 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
finally:
    print("This will always execute.")
```

---

# 8. 기본적인 라이브러리

## NumPy
```python
import numpy as np

# 배열 생성
arr = np.array([1, 2, 3])
print(arr)

# 배열 연산
arr_sum = arr + 10
print(arr_sum)
```

## Pandas
```python
import pandas as pd

# DataFrame 생성
data = {"name": ["Alice", "Bob"], "age": [25, 30]}
df = pd.DataFrame(data)
print(df)

# 특정 컬럼 출력
print(df["name"])
```

---
