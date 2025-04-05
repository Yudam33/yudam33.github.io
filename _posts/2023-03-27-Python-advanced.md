---
title: "Python review advanced"
excerpt: "more for project"

categories: # 카테고리 설정
  - categories1
tags: # 포스트 태그
  - [tag1, tag2]

permalink: /categories1/Pythonadvanced/ # 포스트 URL

toc: true # 우측에 본문 목차 네비게이션 생성
toc_sticky: true # 본문 목차 네비게이션 고정 여부

date: 2025-03-27 # 작성 날짜
last_modified_at: 2025-03-27 # 최종 수정 날짜
---

# Python review advance for project


---

# 1. 고급 함수 및 기능

## 클로저 (Closure)
```python
def outer():
    x = 10
    def inner():
        print(x)
    return inner

closure_func = outer()
closure_func()  # 10
```

## 함수형 프로그래밍 (map, filter, reduce)
```python
# map
result = map(lambda x: x**2, [1, 2, 3])
print(list(result))  # [1, 4, 9]

# filter
result = filter(lambda x: x > 2, [1, 2, 3, 4])
print(list(result))  # [3, 4]

# reduce
from functools import reduce
result = reduce(lambda x, y: x + y, [1, 2, 3, 4])
print(result)  # 10
```

## 람다 함수의 고급 사용
```python
add = lambda x, y, z: x + y + z
print(add(1, 2, 3))  # 6

points = [(1, 2), (4, 5), (2, 3)]
points.sort(key=lambda x: x[1])
print(points)  # [(1, 2), (2, 3), (4, 5)]
```

## 디폴트 인자값
```python
def greet(name="Guest"):
    print(f"Hello, {name}!")

greet()         # Hello, Guest!
greet("Alice")  # Hello, Alice!
```

---

# 2. 제너레이터와 이터레이터

## 제너레이터 (Generator)
```python
def count_up_to(max):
    count = 1
    while count <= max:
        yield count
        count += 1

gen = count_up_to(5)
for number in gen:
    print(number)
```

## 이터레이터 (Iterator)
```python
class Reverse:
    def __init__(self, data):
        self.data = data
        self.index = len(data)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]

rev = Reverse('giraffe')
for char in rev:
    print(char)
```

---

# 3. 데코레이터 (Decorator)

## 기본 데코레이터
```python
def my_decorator(func):
    def wrapper():
        print("Before function")
        func()
        print("After function")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

## 인자와 함께 동작하는 데코레이터
```python
def greet_decorator(func):
    def wrapper(name):
        print("Hello,", name)
        return func(name)
    return wrapper

@greet_decorator
def say_name(name):
    print(f"Your name is {name}")

say_name("Alice")
```

---

# 4. 다중 상속 및 믹스인 (Mixin)

## 다중 상속
```python
class A:
    def method_A(self):
        print("Method A")

class B:
    def method_B(self):
        print("Method B")

class C(A, B):
    def method_C(self):
        print("Method C")

obj = C()
obj.method_A()
obj.method_B()
obj.method_C()
```

## Mixin 클래스
```python
class DatabaseMixin:
    def connect_to_db(self):
        print("Connecting to database...")

class User(DatabaseMixin):
    def get_user(self):
        print("Fetching user data")

user = User()
user.connect_to_db()
user.get_user()
```

---

# 5. 메타클래스 (Metaclass)
```python
class MyMeta(type):
    def __new__(cls, name, bases, dct):
        print(f"Creating class {name}")
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=MyMeta):
    pass

obj = MyClass()
```

---

# 6. 컨텍스트 관리자 (Context Manager)

## with 문을 이용한 파일 관리
```python
with open('file.txt', 'w') as file:
    file.write("Hello, world!")
```

## 커스텀 컨텍스트 관리자
```python
class MyContextManager:
    def __enter__(self):
        print("Entering context...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting context...")

with MyContextManager() as cm:
    print("Inside context")
```

---

# 7. 성능 최적화 및 메모리 관리

## 리스트 대신 제너레이터 사용
```python
numbers = [x**2 for x in range(1000000)]  # 메모리 사용 많음
numbers_gen = (x**2 for x in range(1000000))  # 메모리 효율적
```

## functools.lru_cache로 캐싱
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def expensive_function(n):
    print("Calculating...")
    return n * n

print(expensive_function(5))  # Calculating...
print(expensive_function(5))  # 캐시된 값 사용
```

---

# 8. 고급 라이브러리

## NumPy 고급 기능
```python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
result = arr1 + arr2
print(result)  # [5 7 9]

matrix = np.array([[1, 2], [3, 4]])
print(np.transpose(matrix))
```

## Pandas 고급 사용법
```python
import pandas as pd

arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
df = pd.DataFrame({'data': [1, 2, 3, 4]}, index=arrays)
print(df)

filtered_df = df[df['data'] > 2]
print(filtered_df)
```

## Asyncio를 활용한 비동기 프로그래밍
```python
import asyncio

async def foo():
    print("Start Foo")
    await asyncio.sleep(1)
    print("End Foo")

async def bar():
    print("Start Bar")
    await asyncio.sleep(2)
    print("End Bar")

async def main():
    await asyncio.gather(foo(), bar())

asyncio.run(main())
```

---
