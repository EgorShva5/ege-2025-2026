def f(s,n):
    if s >= 125: return n%2==0
    if n == 0: return 0
    
    table = [f(s+2,n-1),f(s+4,n-1),f(s*2,n-1)]
    
    return all(table) if n%2==0 else any(table)

print('19', [i for i in range(1,125) if f(i,2)])
print('20', [i for i in range(1,125) if not f(i,1) and f(i,3)])
print('21', [i for i in range(1,125) if not f(i,2) and f(i,4)])