import sys
sys.setrecursionlimit(100_000)

def f(n):
    if n == 1: return 1
    elif n > 1: return n*f(n-1)

print((f(87654)-87650*f(87653))/f(87652))