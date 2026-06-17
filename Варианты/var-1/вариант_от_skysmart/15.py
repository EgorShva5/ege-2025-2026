def check_a(a):
    for x in range(10_000):
        if not ( not (x&57 != 0) or ( not (x&38==0) or (x&a != 0) )):
            return 0
    return 1

for a in range(10_000):
    if check_a(a):
        print(a)
        break