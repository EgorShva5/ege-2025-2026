
def alg(n):
    n_t = bin(n)[2:]
    
    if n%2 == 0:
        n_t += '011'
    else:
        n_t += '100'
    
    return int(n_t,2)

for i in range(100):
    if alg(i) < 40:
        print(alg(i),i)