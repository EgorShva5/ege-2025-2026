import sys
sys.setrecursionlimit(100_000)

def f(n):
    if n >= 20: return f(n-4) + 4_620
    if n < 20: return 8*(g(n-12)-21)

def g(n):
    if n >= 384_242: return n/4 + 18
    if n < 384_242: return 12 + g(n+41)

print(f(913))