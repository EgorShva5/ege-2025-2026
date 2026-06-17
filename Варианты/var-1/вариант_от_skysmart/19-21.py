def f(s,n):
    if s >= 99: return n%2==0
    if n == 0: return 0
    
    table = [f(s+2,n-1),f(s*2,n-1)]
    
    return all(table) if n%2==0 else any(table)

print('19', [s for s in range(1,98) if f(s,2)])
print('20', [s for s in range(1,98) if not f(s,1) and f(s,3)])
print('21', [s for s in range(1,98) if not f(s,2) and f(s,4)])

    