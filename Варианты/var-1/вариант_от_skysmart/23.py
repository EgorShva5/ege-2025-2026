def f(n,k):
    if n > k: return 0
    if n == k: return 1
    
    return f(n+1,k) + f(n+2,k) + f(n*3,k)

print(f(1,15)*f(15,27))