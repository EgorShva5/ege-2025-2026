def f(n):
    new_n = bin(n)[2:]
    if n % 3 == 0:
        new_n += new_n[-3:]
    else:
        new_n += bin((3*(n % 3 + 1)))[2:]
    return int(new_n, 2)


m = 0
for n in range(1, 1000):
    r = f(n)
    if r <= 416:
        m = max(m, r)
        
print(m)