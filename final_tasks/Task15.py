'''
p = list(range(66,67+1))
o = list(range(32,125+1))
t = list(range(30,491+1))

nums = [30,32,66,67,125,491]

def check_a(a):
    for x in range(10_000):
        if not ((not x in a) <= ((x in p) or (not x in o) or (not x in t))):
            return 0
    return 1

minimum = 1_000_000_000
for start in nums:
    for end in nums:
        a = list(range(start,end+1))
        
        if check_a(a): minimum = min(minimum, end-start)
print(minimum)'''

'''
for a in range(1,1000):
    for x in range(1,1000):
        for y in range(1,1000):
            if ((78125 != y+4*x) or (a>x) and (a > y)):
                print(a)
                break
'''