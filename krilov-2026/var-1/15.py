def search_dels(n):
    dels = set()
    for i in range(2, int(n**0.5)+1): 
        if n % i == 0:
            dels.add(i)
            dels.add(n//i)
            
    return dels 

a = list(range(3,60))
b = search_dels(177)

def check_y(y):
    dels_y = search_dels(y)
    if len(dels_y) == 0:
        return False
    for x in range(1,10_000):
        if not ((x in dels_y) <= ((x in a) and not (x in b))):
            return False
    return True

for y in range(10_000, 0, -1):
    if check_y(y):
        print(y)
        break
