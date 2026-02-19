for a in range(1,1000):
    for x in range(1,1000):
        for y in range(1,1000):
            if ((78125 != y+4*x) or (a>x) and (a > y)):
                print(a)
                break