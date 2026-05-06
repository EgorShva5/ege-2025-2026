b = 100
c = list(range(b+1))
c[1] = 0

for i in range(2,b):
    if c[i] != 0:
        j = i*2
        
        while j < b:
            c[j] = 0
            j += i

print(c)
        
        
    

'''import math

a, b = 2, 100000
ls = []
st = set()

for i in range(a, b + 1):
    if not i in st:
        ls.append(i)
        for j in range(2 * i, b + 1, i):
            st.add(j)

print(*st, sep='\n')'''


#19
#9 0 0 1 0 0 5 0 3 0 1 1 0 3 2 7 0 1 2

'''n = int(input())
a = list(map(int, input().split()))
stack = []
for i in range(n):
    while len(stack) > 0 and stack[-1] >= i + 1 - a[i]:
        stack.pop()
    stack.append(i+1)
print(len(stack), stack)   '''

#4 4
#1 3 5 8

'''N, D = map(int, input().split())
B = list(map(int, input().split()))
A = [0]
for j in range(1, len(B) + 1):
    A.append(A[j-1] + B[j-1])

i = j = 0
num = 0
while i < N and j < N:
    if A[j] - A[i] < D:
        j += 1
    else:
        num += N - j
        i += 1
print(num)'''

'''n,d = map(int, input().split())
a = list(map(int, input().split()))

i = 0
j = 1

cnt = 0
while i < n and j<n:
    if a[i] + a[j] == 6:
       i += 1
       j += 1
       cnt += 1
    else:
        j += 1

print(cnt)'''
        

'''n, d = map(int, input().split())
a = list(map(int,input().split()))

i = 0
j = 0

summa = 0

while i < n and j < n:
    if a[j]-a[i] >= d:
        summa += n-j
        i+=1
    else:
        j+= 1
        
print(summa)
'''
'''n = int(input())

left = 0
right = 10 ** 18

while right > left+1:
    mid = (left+right)//2
    if mid - mid//13-mid//17 >= n:
        right = mid
    else:
        left = mid
print(left)'''

'''n = int(input())

left = 0
right = n**2

while right > left + 1:
    mid = (left+right)//2
    if mid*mid*mid > n:
        right = mid
    else:
        left = mid

print(left)'''

'''n,x,y = map(int,input().split())

left = 0
right = min(x,y)*(n-1)

while right > left +1:
    mid = (left+right)//2
    duration = mid//x+mid//y
    if duration >= n-1:
        right = duration
    else:
        left = duration

print(right)'''

'''n,k = map(int, input().split())
a = [int(input()) for _ in range(n)]

def find_otrezki(chislo):
    sum_of_otr = 0
    for i in a:
        sum_of_otr += i//chislo
    return sum_of_otr

def bin_search():
    left = 0
    right = sum(a)

    while right > left+1:
        mid = (left+right)//2
        
        if find_otrezki(mid) >= k:
            left = mid
        else:
            right = mid
        
    return left

print(bin_search())
'''
'''n = int(input())
a = list(map(int, input().split()))
m = int(input())
b = list(map(int, input().split()))

def find_chislo(chislo):
    left = -1
    right = len(a)
    
    while right > left + 1:
        mid = (left+right)//2
        if a[mid] > chislo:
            right = mid
        else:
            left = mid
        
    return left

sp = []
for i in b:
    lb = find_chislo(i)
    if lb < len(a) and a[lb] == i:
        sp.append(str(lb+1))
    else:
        sp.append('0')

print(' '.join(sp))'''

'''
n = int(input())
a = list(map(int, input().split()))
m = int(input())
b = list(map(int, input().split()))

def find_enter(chislo, first: bool):
    left = -1
    right = len(a)
    
    while right > left+1:
        mid = (left+right)//2
        
        if (a[mid] >= chislo if first else a[mid] > chislo):
                right = mid
        else:
            left = mid

    return (right if first else left) 


sp = []
for i in b:
    lb = find_enter(i, True)
    rb = find_enter(i,False)
    
    if lb < len(a) and a[lb] == i:
        sp.append(str((rb+lb)//2 + 1))
    else:
        sp.append('0')

print(' '.join(sp))
'''

'''n = int(input())
dels = []
for i in range(1,int(n**(1/2))+1):
    if n % i == 0:
        dels.append(i)
        if i != n // i: dels.append(n//i)
dels.sort()
print(*dels)'''
'''
n, m = map(int, input().split())

graph = [[] for i in range(n + 1)]

for _ in range(m):
    v, u, w = map(int, input().split())
    graph[v].append((u, w))
    graph[u].append((v, w))

min_dist = [10**10] * (n + 1)
for city in range(1, n + 1):
    for i, w in graph[city]:
        if w < min_dist[city]:
            min_dist[city] = w

print(max(min_dist[1:]))
'''

'''
a, b = map(int, input().split())
c = list(map(int, input().split()))

prefix = [0] * (a + 1)
for i in range(1, a + 1):
    prefix[i] = prefix[i - 1] + c[i - 1]

def binarniy_poisk(arr, x):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] <= x:
            left = mid + 1
        else:
            right = mid
    return left

for _ in range(b):
    x, p = map(int, input().split())
    start = prefix[p - 1]
    target = start + x  
    index = binarniy_poisk(prefix, target) 
    print(index - (p - 1)-1)'''




'''
n = int(input())
a = list(map(int, input().split()))

prefix = [0] * (n + 1)
for i in range(1, n + 1):
    prefix[i] = prefix[i - 1] + a[i - 1]

min_prefix = 0
min_pos = 0
max_sum = a[0]
start = 1
end = 1

for i in range(1, n + 1):
    current_sum = prefix[i] - min_prefix
    if current_sum > max_sum:
        max_sum = current_sum
        start = min_pos + 1  # индексация с 1
        end = i

    if prefix[i] < min_prefix:
        min_prefix = prefix[i]
        min_pos = i

print(start, end, max_sum)
'''

'''N = int(input())
dp = [1000000] * (N + 1)
parent = [0] * (N + 1)

dp[1] = 0
for i in range(1, N):
    for nxt in (i + 1, i * 2, i * 3):
        if nxt <= N and dp[nxt] > dp[i] + 1:
            dp[nxt] = dp[i] + 1
            parent[nxt] = i
path = []
cur = N
while cur != 0:
    path.append(cur)
    cur = parent[cur]
path.reverse()

print(dp[N])
print(*path)
'''


'''n = int(input())
a = list(map(int, input().split()))

result = 0
prefix_sum = 0

for x in a:
    result += x * prefix_sum 
    prefix_sum += x         

print(result)'''
