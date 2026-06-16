
#№ 29347 Открытый вариант 2026 (Уровень: Базовый)

'''
b = list(range(22,41))
c = list(range(32,51))

nums = (22,40,32,50)

def check_a(a):
    for x in range(100):
        if not ((x in a) or ((x in b) == (x in c))):
            return 0 
    return 1

mini = float('inf')
for start in nums:
    for end in nums:
        a = list(range(start, end+1))
        
        if a != []:
            if check_a(a): mini = min(mini, end-start)
print(mini)
'''

'''
p = list(range(7,15))
q = list(range(9,12))

nums = [7,9,11,14]

def check_a(a):
    for x in range(10_000):
        if not(not((x in p) and (x in q)) or (x in a)):
            return 0
    return 1

maxi = float('inf')
for start in nums:
    for end in nums:
        a = list(range(start, end+1))
        
        if a != [] and check_a(a): 
            print(a)
            maxi = min(maxi, end-start)
print(maxi)
'''
        
'''
k = list(range(20,62))
u = list(range(37,81))

nums = [20,37,61,81]

def check_a(a):
    for x in range(10_000):
        if not ((x in a) or (x in u) or not (x in k)):
            return 0
    return 1

minimum = float('inf')
for start in nums:
    for end in nums:
            a = list(range(start,end+1))
            if check_a(a): minimum = min(minimum, end-start)   

print(minimum)
'''

'''
p = list(range(11,46))
q = list(range(32,65))

nums = [11,32,45,64]

def check_a(a):
    for x in range(10_000):
        if not(not (x in p) or not ((x in q) and not (x in a) or (not (x in p)))):
            return 0
    return 1

minimum = 1_000_000_000
for start in nums:
    for end in nums:
        a = list(range(start,end+1))
        if check_a(a): minimum = min(minimum, end-start)
print(minimum)
'''

'''
def check_a(a):
    for x in range(100_000): 
        if not (x&51==0 or (x&a!=0 or x&25!=0)):
            return 0
    return 1

for a in range(1_000_000):
    print(a, check_a(a))
    if check_a(a):
        print(a)
        break
'''

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