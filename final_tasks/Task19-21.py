#	№ 28704 (Уровень: Базовый)

'''def f(s1,s2,n):
    if s1*s2 >= 516: return n%2==0
    if n ==0: return 0
    
    table = [f(s1+3,s2,n-1),f(s1,s2+3,n-1),f(s1+13,s2,n-1),f(s1,s2+13,n-1)]
    
    return all(table) if n%2==0 else any(table)

print('19', [i for i in range(1,74) if f(7,i,2)])
print('20', [i for i in range(1,74) if not f(7,i,1) and f(7,i,3)])
print('20', [i for i in range(1,74) if not f(7,i,2) and f(7,i,4)])
'''

'''def f(s1,s2,n):
    if s1+s2 >= 102: return n%2==0
    if n == 0: return 0
    
    table = [f(s1+1,s2,n-1),f(s1,s2+1,n-1),f(s1*2,s2,n-1),f(s1,s2*3,n-1)]
    
    return all(table) if n%2==0 else any(table)

print('19', [i for i in range(1,102) if f(i,24,2)])
print('20', [i for i in range(1,102) if not f(i,24,1) and f(i,24,3)])
print('21', [i for i in range(1,102) if not f(i,24,2) and f(i,24,4)])
'''

#	№ 28704 (Уровень: Базовый)

'''def f(s1,s2,n):
    if s1+s2 >= 516: return n%2==0

'''
#№ 29351 Открытый вариант 2026 (Уровень: Базовый)
'''def f(s1,s2,n):
    if s1+s2 >= 154: return n%2==0
    if n==0: return 0
    
    table = [f(s1+4,s2,n-1),f(s1,s2+4,n-1),f(s1*3,s2,n-1),f(s1,s2*3,n-1)]
    
    #19 задание - return any(table)
    return all(table) if n%2==0 else any(table)

print('19', [i for i in range(1,143) if f(11,i,2)])
print('20', [i for i in range(1,143) if not f(11,i,1) and f(11,i,3)])
print('21', [i for i in range(1,143) if not f(11,i,2) and f(11,i,4)])'''

'''
def f(s1,s2,n):
    if s1+s2 >= 211: return n % 2 == 0
    if n == 0: return 0
    
    table = [f(s1+1,s2,n-1),f(s1*2,s2,n-1),f(s1,s2+1,n-1),f(s1,s2*2,n-1)]
    
    return all(table) if n%2==0 else any(table)

print('19', [i for i in range(1,193) if f(17,i,2)])
print('20', [i for i in range(1,193) if not f(17,i,1) and f(17,i,3)])
print('21', [i for i in range(1,193) if not f(17,i,2) and f(17,i,4)])
'''

'''def f(s,n):
    if s >= 125: return n%2==0
    if n == 0: return 0
    
    table = [f(s+2,n-1),f(s+4,n-1),f(s*2,n-1)]
    
    return all(table) if n%2==0 else any(table)

print('19', [i for i in range(1,125) if f(i,2)])
print('20', [i for i in range(1,125) if not f(i,1) and f(i,3)])
print('21', [i for i in range(1,125) if not f(i,2) and f(i,4)])
'''