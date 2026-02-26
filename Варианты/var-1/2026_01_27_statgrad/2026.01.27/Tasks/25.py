def find_dels(n):
    dels = set()
    
    for i in range(1, int(n**0.5)+1):
        if n%i == 0: 
            dels.add(i)
            dels.add(n//i)
        
    return dels

def check_chislo(n):
    dels = find_dels(n)
    summa_dels = len(dels)
    if (n-summa_dels) % 23 == 0:
        return 1
    else: return 0

chisla = []
for i in range(999_999_999, 1_000_000, -1):
    if check_chislo(i):
        chisla.append(i)

        if len(chisla) == 5: break

print(sorted(chisla)[:5])