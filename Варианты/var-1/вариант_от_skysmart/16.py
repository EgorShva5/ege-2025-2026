import sys

sys.setrecursionlimit(10_000)

def f(n):
    if n <= 1: return 1
    if n > 1 and n%2==0: return n/2 + f(n-3)
    if n > 1 and n%2!=0: return n + f(n+1)

print(f(2000)-f(1500))