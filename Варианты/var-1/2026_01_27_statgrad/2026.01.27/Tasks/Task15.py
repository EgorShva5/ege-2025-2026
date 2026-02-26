def f(a):
    for x in range(1,100_000):
        for y in range(1, 100_000):
            if not ((y<a) and (x<a) or (89_241<(5*y+x))): 
                return 0
    return 1

for a in range(1,100_000):
    if f(a): print(a)
            
