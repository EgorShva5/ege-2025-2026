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
