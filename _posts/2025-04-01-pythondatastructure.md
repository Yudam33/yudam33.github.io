---
title: "Python data structure"
excerpt: "Array,stack,queue,tree,graph implementation"

categories: # 카테고리 설정
  - python
tags: # 포스트 태그
  - [tag1, tag2]

permalink: /categories1/datastructure/ # 포스트 URL

toc: true # 우측에 본문 목차 네비게이션 생성
toc_sticky: true # 본문 목차 네비게이션 고정 여부

date: 2025-04-01 # 작성 날짜
last_modified_at: 2025-04-01 # 최종 수정 날짜
---

# Python 자료구조

## 1️⃣ 배열 (Array)
### ✅ 리스트 생성 및 조작
```python
arr = [1, 2, 3, 4, 5]
arr.append(6)  # 요소 추가
arr.pop()  # 마지막 요소 제거
arr.insert(2, 10)  # 특정 위치에 요소 삽입
arr.remove(3)  # 특정 값 제거
arr.sort()  # 정렬
arr.reverse()  # 역순 정렬
```

### ✅ 리스트 컴프리헨션
```python
squared = [x**2 for x in range(10)]  # [0, 1, 4, 9, ..., 81]
```

---

## 2️⃣ 해시 테이블 (Hash Table)
### ✅ 딕셔너리 기본 조작
```python
hash_map = {"a": 1, "b": 2, "c": 3}
print(hash_map["a"])  # 1
hash_map["d"] = 4  # 새로운 키 추가
del hash_map["b"]  # 키 삭제
```

### ✅ 키 존재 여부 확인
```python
if "a" in hash_map:
    print("Key exists!")
```

---

## 3️⃣ 연결 리스트 (Linked List)
### ✅ 단일 연결 리스트 구현
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        if not self.head:
            self.head = Node(data)
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = Node(data)
```

---

## 4️⃣ 스택 (Stack)
### ✅ 리스트를 이용한 스택 구현
```python
stack = []
stack.append(1)  # push
stack.append(2)
print(stack.pop())  # pop -> 2
```

### ✅ collections 모듈 활용
```python
from collections import deque
stack = deque()
stack.append(1)
stack.append(2)
print(stack.pop())
```

---

## 5️⃣ 큐 (Queue)
### ✅ collections.deque 사용
```python
from collections import deque
queue = deque()
queue.append(1)  # enqueue
queue.append(2)
print(queue.popleft())  # dequeue -> 1
```

---

## 6️⃣ 힙 (Heap)
### ✅ 최소 힙
```python
import heapq
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)
print(heapq.heappop(heap))  # 1 (최소값 제거)
```

### ✅ 최대 힙
```python
heapq.heappush(heap, -3)  # 음수로 저장하여 최대 힙처럼 사용
```

---

## 7️⃣ 트리 (Tree)
### ✅ 이진 트리 기본 구조
```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
```

### ✅ 이진 트리 순회 (DFS)
```python
def inorder_traversal(node):
    if node:
        inorder_traversal(node.left)
        print(node.value)
        inorder_traversal(node.right)
```

---

## 8️⃣ 그래프 (Graph)
### ✅ 인접 리스트 방식 그래프 구현
```python
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}
```

### ✅ BFS (너비 우선 탐색)
```python
from collections import deque
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node)
            visited.add(node)
            queue.extend(graph[node])
bfs(graph, 'A')
```

### ✅ DFS (깊이 우선 탐색)
```python
def dfs(graph, node, visited=set()):
    if node not in visited:
        print(node)
        visited.add(node)
        for neighbor in graph[node]:
            dfs(graph, neighbor, visited)
dfs(graph, 'A')
```

---

이 Python Cheat Sheet를 활용하면 자료구조 문제를 풀 때 빠르게 참고할 수 있어! 🚀

