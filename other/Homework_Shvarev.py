'''t = open('task2.txt')

sp = [int(i) for i in t]
chis = 0
maxim = 0

for i in range(1,len(sp)):
    sp2 = [sp[i-1],sp[i]]

    if (sp2[0] % 3 == 0 or sp2[1] %3 == 0):
        chis += 1
        maxim = max(maxim, sum(sp2))

print(chis, maxim)
'''
'''def F(n):
    if n == 1:
        return 1
    if n > 1:
        return F(n-1) * n
print(F(5))'''

'''ch = '8'*68

while ('222' in ch) or ('888' in ch):
    if '222' in ch:
        ch = ch.replace('222', '8', 1)
    else:
        ch = ch.replace('888', '2', 1)
    
    print(ch)
    
print(ch)'''

'''def f(n):
    new_n = bin(n)[2:]
    ch = new_n.count('1')
    if ch%2==0: new_n += '00'
    else: new_n += '11'

    return int(new_n,2)

for i in range(100):
    chislo = f(i)
    if chislo > 114:
        print(chislo)
        break'''

'''def to_6(n):
    result = ''
    
    while n>0:
        result = str(n%6) + result
        n //=6
    
    return result
    
print(to_6(12))'''

'''def f(n):
    new_n = bin(n)[2:]
    summa = new_n.count('1') % 2
    new_n = new_n + str(summa)
   # print(new_n)
    
    summa = new_n.count('1')%2
    new_n = new_n + str(summa)
    #print(new_n + ' | ' + str(int(new_n,2)) + '\n\n\n')
    
    return int(new_n,2)

for r in range(1,100):
    if f(r) > 77:
        print(r)
        break'''
    

'''def f(a,b,m):
    if a+b >= 82: return m%2==0
    
    if m == 0: return 0
    h = [f(a+1,b,m-1), f(a,b+1,m-1), f(a*4,b,m-1), f(a,b*4,m-1)]
    
    return any(h) if (m-1)%2==0 else all(h)

print([s for s in range(1,78) if f(4,s,2)])
print([s for s in range(1,78) if not f(4,s,1) and f(4,s,3)])
print([s for s in range(1,78) if not f(4,s,2) and f(4,s,4)])'''
'''text = open('task.TXT')

ch = 0

for i in text:
    string = i.split()
    
    if len(set(string)) == 5:
        p = ''
        for b in string:
            if string.count(b) == 2:
                p = b
                break
        
        string = list(map(int, string))
        
        sr_ar_p = int(p)
        sr_ar_n_p = (sum(string)-int(p)*2) / 4 

        if sr_ar_p < sr_ar_n_p:
            ch += 1
print(ch)
'''
'''from itertools import *

b = 0

for i in product('БОРИС', repeat=6):
    if i.count('Б') == 1 and i.count('Р') == 1 and (i.count('С') == 1 or i.count('С') == 0):
        b+=1

print(b)'''


'''from turtle import *

screensize(5000,5000)
tracer(0)

a = 3

down()
color('black')
forward(100*a)

update()
done()'''

'''from itertools import *

def f(x,y,z,w):
    return (((y <= z) or (not x and w)) == (w == z))

for a1,a2,a3,a4 in product([0,1], repeat=4):
    table = [(a1,1,0,0),(0,0,0,1),(0,1,a3,a4)]
    
    if len(table) == len(set(table)):
        for p in permutations('xyzw'):
            if [f(**dict(zip(p,r))) for r in table] == [1,1,1]:
                print(p)
'''

'''a, b, c = map(int, input().split())
ch = input()

for _ in range(c):
    d, e, z = input().split()
    e = int(e) - 1
    
    if d == '?':
        z = int(z)

        result = 0
        for i in range(e, z):
            result = (result * 10 + (ord(ch[i]) - ord('0'))) % b
        print(result)
        
    elif d == '!':
        ch = ch[:e] + z + ch[e + 1:]'''
        
'''
a, b, c = map(int, input().split())
ch = list(input())

for _ in range(c):
    d, e, z = input().split()
    e = int(e) - 1
    if d == '?':
        z = int(z)
        result = 0
        for i in range(e, z):
            result = (result * 10 + int(ch[i])) % b
        print(result)
        
    elif d == '!':
        ch[e] = z'''

'''a,b,c = map(int, input().split())
ch = input()

for i in range(c):
    d,e,z = input().split()
    if d == '?':
        new_ch = int(ch[int(e)-1:int(z)])
        #print(new_ch,
        print(new_ch%b)
        
    elif d == '!':
        ch = ch[:int(e)-1] + z + ch[int(e):] 
#        print(ch)'''
        

'''n = int(input())
d1, s1, p1 = int(input()), int(input()), int(input())
d2, s2, p2 = int(input()), int(input()), int(input())

# Проверяем, подходят ли рейсы по количеству мест
bus1_ok = s1 >= n
bus2_ok = s2 >= n

if not bus1_ok and not bus2_ok:
    print(-1)
else:
    # Выбираем рейсы по приоритетам: поздний день → дешевая цена
    candidates = []
    if bus1_ok:
        candidates.append((d1, p1, 1))
    if bus2_ok:
        candidates.append((d2, p2, 2))
    
    # Сортируем: сначала по дню (поздние выше), затем по цене (дешевые выше)
    candidates.sort(key=lambda x: (-x[0], x[1]))
    
    # Лучший вариант
    best_day, best_price, best_num = candidates[0]
    
    # Собираем все рейсы с такими же днем и ценой
    result_nums = [num for d, p, num in candidates if d == best_day and p == best_price]
    
    # Выводим минимальную стоимость и номера рейсов
    print(best_price * n)
    for num in sorted(result_nums):
        print(num)'''
'''a = int(input())
b = int(input())

new_a = (a - 1) // 4 + 1
b1 = (b - 2 * (new_a - 1) - 1) / 4
b2 = (b - 2 * (new_a - 1) - 2) / 4

if b1.is_integer() and b1 > 0:
    print(int(b1))
elif b2.is_integer() and b2 > 0:
    print(int(b2))'''

'''a = int(input())
b = int(input())

maxim = b
chis = int(maxim%6)
sec_n = 0

if maxim % 2 == 0:
    sec_n = maxim -1
else: sec_n = maxim + 1

while True:
    maxim += 6
    if maxim % 2 == 0:
        sec_n = maxim -1
    else: sec_n = maxim + 1
    
    if maxim % 6 == 0 or sec_n % 6 == 0:
        break
    
    print(sec_n%6, maxim%6)
    
    chis+=1

    
print(chis)
'''


'''n = int(input())
a = list(map(int, input().split()))

prefix_mod = [0] * (n + 1)
pos = dict()
pos[0] = 0  

for i in range(1, n + 1):
    prefix_mod[i] = (prefix_mod[i - 1] + a[i - 1]) % n
    if prefix_mod[i] in pos:
        l = pos[prefix_mod[i]] + 1
        r = i
        print(l, r)
        break
    else:
        pos[prefix_mod[i]] = i
else:
    print(-1)
'''
'''
list_keys= {
    'a':2,'b': 3,'c': 4, #abc
    'd':2, 'e':3,'f':4, #def
    'g':2, 'h':3,'i':4, #GHI
    'j':2, 'k':3,'l':4, #JKL
    'm':2, 'n':3,'o':4, #MNO
    'p':2, 'q':3,'r':4, 's':5, #PQRS
    't':2, 'u':3,'v':4, #TUV
    'w':2, 'x':3,'y':4, 'z':5
}

list_2 = ('abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz')

s = input()
t = int(input())
mult = int(input())

all_sum = 0

prev_k = 'hjkjlkjjjlkjljkjkhuijiuy'

for i in s:
    step = list_keys[i]-1
    all_sum += step*mult
    
    for b in list_2:
        if prev_k in b and i in b:
            
            all_sum += t
            
            break
    
    prev_k = i
    
print(all_sum)'''

'''
n = int(input())
a = list(map(int,input().split()))

def chis_power(osnovanie, power):
    chislo = 1
    for i in range(power):
        chislo *= osnovanie
    return chislo

lower_chis = sum(a)

all_sum = lower_chis * n
for i in range(n):
    print(str(a[i]))
    chislo = len(str(a[i]))
    all_sum += chis_power(10, chislo)*lower_chis

print(all_sum)

'''

'''import heapq

n, k, t = map(int, input().split())
c = list(map(int, input().split()))
a = list(map(int, input().split()))

free = c  
heap = [] 

ispolsov_count = [0] * n  

for i in a:
    while heap and heap[0][0] <= i:
        avt_type = heapq.heappop(heap)[1]
        #print(avt_type)
        free[avt_type] += 1
    
    vibr_avt_t = -10000000
    for b in range(n):
        if free[b] > 0:
            vibr_avt_t = b
            #print(vibr_avt_t)
            break
        
    #print(vibr_avt_t)
    if vibr_avt_t != -1:
        free[vibr_avt_t] -= 1
        ispolsov_count[vibr_avt_t] += 1
        heapq.heappush(heap, (i + t, vibr_avt_t))
    
    #print(i,free,heap)

print(' '.join(str(i) for i in ispolsov_count))'''


'''n,k,t = map(int, input().split())
n = list(map(int,input().split()))
k = list(map(int,input().split()))
'''

'''import math

n = int(input())
spisok = list(map(int, input().split()))

spis_chetn = [0] 
for x in spisok:
    spis_chetn.append((spis_chetn[-1] + x) % 2) 
        
# Т.к чётное + чётное = четное, нечетное+нечетное = чётное, нечётное + чётное = нечётное.
# c n k = n! / k!(n-k)! // c sp_chet 2 = sp_shet! / 2 (sp_chet-2)!

chet_chis = spis_chetn.count(0) 
nechet_chis = spis_chetn.count(1)

summa1 = math.factorial(chet_chis) / (2*math.factorial(abs(chet_chis-2)))
summa2 = math.factorial(nechet_chis) / (2*math.factorial(abs(nechet_chis-2)))

print(int(summa1+summa2))
'''


'''n = int(input())
spisok = list(map(int, input().split()))

spisok_chetn = [0]
for x in spisok:
    spisok_chetn.append((spisok_chetn[-1] + x) % 2)

# Т.к чётное + чётное = четное, нечетное+нечетное = чётное, нечётное + чётное = нечётное.
# c n k = n! / k!(n-k)! // c sp_chet 2 = sp_shet! / 2 (sp_chet-2)! == chet_chis * (chet_chis - 1) // 2  

chet_chis = spisok_chetn.count(0)  
nechet_chis = spisok_chetn.count(1)

summa1 = chet_chis * (chet_chis - 1) // 2  
summa2 = nechet_chis * (nechet_chis - 1) // 2
print(summa1 + summa2)'''


'''
n = int(input())
a = input().split()

nums = list(map(int, a))
lengths = list(map(len, a))

power_sum = sum(10 ** l for l in lengths)
total_sum = sum(nums)

ispolsov_count = 0
for x in nums:
    ispolsov_count += x * power_sum
ispolsov_count += total_sum * n

print(ispolsov_count)
'''

'''from itertools import * 

sum = 0
n = int(input())
a = list(map(int, input().split()))

for i in product(a, repeat=2):
    print(i)
    sum += i

print(sum)'''

'''F = [1,1]
F.extend([0]*1000000)

n = int(input())

def f(n):
    if F[n] == 0:
        F[n] = f(n-1) + f(n-2)
    return F[n]

print(f(n-1))
'''
'''i = {}

a = int(input())

def get_key(d, value):
    for k, v in d.items():
        if v == value:
            return k

for i in range(a):
    b = input().split()
    i[int(str(b[0][:-1]))] = [b[1],b[2],b[3]]

b = int(input())

for i in range(b):
    c = input().split()
    i[int(str(c[0][:-1]))] = [c[1],c[2],c[3]]

c = int(input())
for i in range(c):
    d = input().split()
    i[int(str(d[0][:-1]))] = [d[1],d[2],d[3]]

i = dict(sorted(i.items()))

#name: id, message
users = {}
for i in i:
    sp_d = i[i]
    if sp_d[0] == 'REG':
        if sp_d[2] not in users.keys():
            users[sp_d[2]] = [sp_d[1], {}]
           # print('REG', users)
    elif sp_d[0] == 'CHANGE':
        if sp_d[2] not in users.keys():
            #print('BU', users)
            
            #print('info',users[sp_d[1][0]], users[sp_d[1]][1])
            users[sp_d[2]] = [users[sp_d[1]][0],users[sp_d[1]][1]]
            old = sp_d[1]
            new = sp_d[2]
            #print('ЮЗЗ',sp_d[1], sp_d[2])
            del users[sp_d[1]]
            
            
            for i in users:
                if i != new:
                    users[i][1][new] = users[i][1][old]
                    del users[i][1][old]
                    #print('OldUs',users[i][1][old])
            #print(users, ' g', users[sp_d[2]])
    else:
        try:
            users[sp_d[1]][1][sp_d[2]] = users[sp_d[1]][1][sp_d[2]] + 1
        except:
            users[sp_d[1]][1][sp_d[2]] = 1
        #print('SM',users)
            
print(len(users.keys()))

#print(users)
users = dict(sorted(users.items(), reverse=True))

#print(users)
for i in users:
    total = 0
    
    leaderboard_users = dict(sorted(users[i][1].items()))
    
    #print(sorted(leaderboard_users.values(),reverse=True))
    sc_l = sorted(leaderboard_users.values(),reverse=True)
    #print(get_key(leaderboard_users,sc_l[0]))
    
    id = {}
    id[i] = users[i][1]
    
    
    k = list(leaderboard_users.keys())[0]
    v= str(list(leaderboard_users.values())[0])
    
    for b in users[i][1]:
        total += users[i][1][b]
    
    username = get_key(leaderboard_users,sc_l[0])
    
    #print(users)
    print(str(users[i][0]),'SENT',str(total),'TOP',v,'TO',users[username][0])
'''
#print(i)

'''a, b = map(int, input().split())
c = list(map(int, input().split()))

tup = []
for i in range(len(c)):
    new_sp = c[i:i+b]
    if len(new_sp) < b:
        break
    
    if len(new_sp) == len(set(new_sp)):
        tup.append((new_sp[0],new_sp[-1]))
    
if len(tup) != 0:
    tup.sort(key=lambda x: x[0], reverse=True)
    print(*tup[0])
else:
    print('-1')'''



'''from collections import deque

n, k = map(int, input().split())
a = list(map(int, input().split()))

dq = deque()
for i in range(n):
    while dq and dq[0] <= i - k:
        dq.popleft()

    while dq and a[dq[-1]] >= a[i]:
        dq.pop()
    
    dq.append(i)
    
    if i >= k - 1:
        print(a[dq[0]])'''

'''def min_elements_to_disable(n, k, a):
    elemnts = {}
    for num in a:
        elemnts[num] = elemnts.get(num, 0) + 1

    disabled = 0
    without_povtorenia = set()

    for num in list(elemnts.keys()):
        if num in without_povtorenia:
            continue
        ostatok = k - num

        if ostatok not in elemnts:
            continue

        if num == ostatok:
            disabled += elemnts[num] - 1
        else:
            disabled += min(elemnts[num], elemnts[ostatok])
            without_povtorenia.add(ostatok)
        without_povtorenia.add(num)

    return disabled

a, b = map(int,input().split())
c = list(map(int,input().split()))

print(min_elements_to_disable(a,b,c))'''

'''a = int(input())

stack = []

for i in range(a):
    b = tuple(map(int, input().split()))
    
    if b[0] == 1: stack.append(b[1])
    elif b[0] == 2: stack.pop()
    else: print(min(stack))'''
    
'''
n = int(input())
spisok = list(map(int, input().split()))

last_pos = {}
ispolsov_count = []

for i, val in enumerate(spisok):
    if val in last_pos:
        ispolsov_count.append(i - last_pos[val])
    else:
        ispolsov_count.append(-1)
    last_pos[val] = i

print(*ispolsov_count)'''


'''d = {}

n = int(input())

for i in range(n):
    a = input().split()
    
    if a[0] == '1':
        try:
            d[a[1]] = d[a[1]] + int(a[2])
        except:
            d[a[1]] = 0 + int(a[2])
    else:
        try:
            print(d[a[1]])
        except:
            print('ERROR')
'''
'''s = set()

n = int(input())
for i in range(n):
    a = input().split()
    if a[0] == '+':
        s.add(int(a[1]))
    elif a[0] == '-': 
        s.discard(int(a[1]))
    else:
        print('1') if int(a[1]) in s else print('0')'''

'''from heaheap import heappop, heappush

heap = []

n = int(input())
for i in range(n):
    a = input().split()
    
    if a[0] == '0':
        heappush(heap, -int(a[1]))
    else:
        print(-heappop(heap))
        '''

'''n = int(input())

b = []
delited = []

def push(i):
    b.append(i)

def pop():
    delit = b.pop(0)
    delited.append(delit)

for i in range(n):
    a = tuple(input().split())
    a1 = a[0]
    if a1 == '+':
        push(a[1])
    if a1 == '-':
        pop()

print(*delited)'''

'''a = int(input())
string = ''

for i in sorted(map(int, input().split()),reverse=True):
    string += str(i)

print(string)'''

"""a = int(input())
books=list(map(int, input().split()))
m,k=map(int,input().split())
nums =list(map(int,input().split()))
"""

"""
a,r = map(int, input().split())
b = list(map(int,input().split()))

ans = 0

i, j = 0

for i in range(a):
    while j < a:
"""

"""
a = int(input())
books=list(map(int, input().split()))
m,k=map(int,input().split())
nums =list(map(int,input().split()))

def find_book(start):
    s_p = start
    max_num = books[start]
    cur_k = 0
    last_num = books[start]
    
    while s_p >= 0:
        s_p-=1
        
        if books[s_p] == last_num: cur_k += 1
        else: cur_k = 1
        
        last_num = books[s_p]
        
        if cur_k > k-1:
            break
        
        if books[s_p] > max_num:
            break
        
        #print(cur_k,k,books[s_p])
        
    return s_p+1

for i in nums:
    print(find_book(i-1))

"""

"""n, act_rost = map(int, input().split())
all_apples_info = [tuple(map(int, input().split())) + (i+1,) for i in range(n)]

group_big = [x for x in all_apples_info if x[1] > x[0]]
group_small = [x for x in all_apples_info if x[1] <= x[0]]

group_big.sort(key=lambda x: x[0])
group_small.sort(key=lambda x: x[1], reverse=True)

all_sp = group_big + group_small

for a, b, idx in all_sp:
    if act_rost - a <= 0:
        print(-1)
        break
    act_rost = act_rost - a + b
#    print(a,b,idx,act_rost)
else:
    string = ''
    for x in all_sp:
        string += str(x[2]) + " "
    print(string)

#print(all_sp)
"""

"""
n, act_rost = map(int, input().split())

slov = {}
sum_plus = 0
sum_minus = 0

for i in range(n):
    minus_val, plus_val = map(int, input().split())
    sum_plus += plus_val
    sum_minus += minus_val
    slov[minus_val] = i + 1

if sum_plus > sum_minus:
    string = ''
    for key in sorted(slov):
        string += str(slov[key]) + ' '
    print(string)
else:
    print('-1')
"""

"""a = list(map(int,input().split()))

act_rost = a[1]

plus = []
minus = []
slov = {}

print('fghg')
for i in range(a[0]):
    b = list(map(int,input().split()))
    plus.append(b[1])
    minus.append(b[0])
    slov[b[0]] = i+1
    print('fghfgh')

print('fgh')
if sum(plus) > sum(minus):
    new_sl = sorted(slov)
    for i in new_sl:
        print(slov[i])
else:
    print('-1')"""


"""
a = list(map(int,input().split()))
b = sorted(map(int,input().split()))

ch = 0
for i in range(a[0]):
    j = a[0]-1
    while j > i:
        if b[j] - b[i] > a[1]:
            ch += 1
        else:
            break
        j -= 1
        
print(ch)
        
"""
    
    

"""
a = list(map(int, input().split()))
b = sorted(map(int, input().split()))
c = sorted(map(int, input().split()))

i = j = 0
chis = abs(b[0] - c[0])

while i < len(b) and j < len(c):
    diff = abs(b[i] - c[j])
    if diff < chis:
        chis = diff

    if b[i] < c[j]:
        i += 1
    else:
        j += 1

print(chis)"""

"""
a = list(map(int,input().split()))
b = sorted(map(int,input().split()))
c = sorted(map(int,input().split()))

j = 0

chis = abs(a[0]-b[0])
for i in range(a[0]-1):
    while j<a[1] and c[j] < b[i]:
        j += 1
    
    if j > 0:
        chis = min(chis, abs(b[i]-c[j-1]))
    if j < a[1]:
        chis = min(chis, abs(c[j]-b[i]))
    
print(chis)
"""

"""
n = list(map(int, input().split()))
k = sorted(map(int, input().split()))

summa = 0
ch = 0

for i in k:
    if summa + i > n[0]:
        break
    summa += i
    ch += 1

print(ch)
"""

"""
a, k, b, m, x = map(int, input().split())

def NOK(a, b):
    m = a * b
    while a != 0 and b != 0:
        if a > b:
            a %= b
        else:
            b %= a
    return m // (a + b)
l = NOK(k, m)
#print(l)

def chop_tree(d):
    dima_d = d - d // k
    feda_d = d - d // m
    oba_otdixayout = d // l
    #print(dima_d,feda_d,oba_otdixayout)
    return a * dima_d + b * feda_d - (a + b) * oba_otdixayout

left =1
right = 1000000000000000
while left < right:
    mid = (left + right) // 2
    #print(mid)
    if chop_tree(mid) >= x:
        right = mid
    else:
        left = mid + 1

print(left)

"""

"""
import math

C = float(input())

left, right = 0, C
for i in range(40):
    middle = (left + right) / 2
    #print(left,right)
    if middle * middle + math.sqrt(middle) <= C:
        left = middle
    else:
        right = middle

print(right)

"""

"""
from math import sqrt

Vp, Vf = map(int, input().split())
a = float(input())

def i(x):
    return sqrt(x*x + (a - 1)**2) / Vp + sqrt((1 - x)**2 + a*a) / Vf

left, right = 0.0, 1.0
for _ in range(100):  
    m1 = left + (right - left) / 3
    m2 = right - (right - left) / 3
    if i(m1) > i(m2):
        left = m1
    else:
        right = m2

print(f"{(left + right) / 2:.9f}")
"""

"""import math

a = [ int(i) for i in input().split() ]
b = float(input())

def func(x):
    return math.dist((0,1),(x,b))/a[0] + math.dist((x,b),(1,0))/a[1]
left = 0
right = 1

for i in range(40):
    min1 = left+(right-left)/3
    min2 = right-(right-left)/3
    
    #print(left,right)
    if func(min1) > func(min2):
        left = min1
    else:
        right = min2
        
print(left)"""


"""
n = int(input())
sp = list(map(int, input().split()))
sp.sort()

k = int(input())

def minim(sp, x):
    left = 0
    right = len(sp)
    
    while left < right:
        mid = (left + right) // 2
        if sp[mid] < x:
            left = mid + 1
        else:
            right = mid
    return left

def maxim(sp, x):
    left, right = 0, len(sp)
    while left < right:
        mid = (left + right) // 2
        if sp[mid] <= x:
            left = mid + 1
        else:
            right = mid
    return left

ch = []

for _ in range(k):
    l, r = map(int, input().split())
    left = minim(sp, l)
    right = maxim(sp, r)
    ch.append(abs(left-right))

ch1 = [str(i) for i in ch]

print(' '.join(ch1))
"""

"""
5
10 1 10 3 4
4
1 10
2 9
3 4
2 2

5 2 2 0 
"""

#Тернарный поиск
"""
C = float(input().strip())

def f(x):
    return x * x + x ** 0.5

left, right = 0.0, (C ** 0.5) + 1  # Добавляем 1 для надежности

# Бинарный поиск с высокой точностью
for _ in range(100):  # 100 итераций достаточно для точности ~1e-30
    mid = (left + right) / 2
    if f(mid) < C:
        left = mid
    else:
        right = mid

print(f"{left:.20f}")
"""

#Бинарный поиск
"""
a = [ int(i) for i in input().split() ]
b = sorted([ int(i) for i in input().split() ])
c = [ int(i) for i in input().split() ]

#m1, m2 = a[0], a[1]

#print(b)

def BinSearch(value):
    left = 0
    #print(b, len(b))
    right = len(b)-1
    
    best = b[0]
    
    while left <= right:
        mid = (left+right)//2
        
        if abs(b[mid] - value) < abs(best - value) or (abs(b[mid] - value) == abs(best - value) and b[mid] < best):
            best = b[mid]
        
        #print(b[mid])
        #print(b[mid], c[0])
        if b[mid] == value:
            return b[mid]
        elif b[mid] < value:
            left = mid +1
            #print('lf')
        else: 
            right = mid -1
            #print('ytn')

        #print(left,right)
    return best
        #print(left,right)
        
#print(BinSearch(2))
for i in c:
    print(BinSearch(i))
   
"""

"""  
     5 5
1 3 5 7 9
2 4 8 1 6
"""
#Дипломы
"""
import math

a = [int(i) for i in input().split()]

wid = a[0]
hei = a[1]
num = a[2]

left = 0
right = max(wid, hei) * int(math.ceil(num ** 0.5)) * 2

while left < right:
    mid = (left + right) // 2
    value = (mid // wid) * (mid // hei)
    
    if value >= num:
        right = mid
    else:
        left = mid + 1

print(left)

"""

"""
import math

a = [int(i) for i in input().split()]

wid = a[0]
hei = a[1]
num = a[2]

s = wid*hei
s2 = s

while True:
    print(int(s2**(1/2)),int(math.ceil(s2**(1/2))))
    
    if int(s2**(1/2)) == int(math.ceil(s2**(1/2))):
        b = s2**(1/2)
        
        summa = int(b/wid) * int(b/hei)


        print(summa,b)
        if summa >= num:
            print(int(b))
            break
    
    s2 += 1
    
""" 

"""
a = input()
b = input()

wood = int(a.split()[0])
iron = int(a.split()[1])

wood_p = int(b.split()[0])
iron_p = int(b.split()[1])

n = wood + iron

def sum_p(k):
    if k <= wood:
        return k * wood_p
    else:
        return wood * wood_p + (k - wood) * iron_p

maxim= wood * wood_p + iron * iron_p
mid_1 = maxim / 2

left, right = 0, n
best_k = 0
best_diff = maxim

while left <= right:
    mid = (left + right) // 2
    tom = sum_p(mid)
    ghek = maxim - tom
    diff = abs(tom - ghek)

    #print(diff,tom,ghek,mid)
    if diff < best_diff:
        best_diff = diff
        best_k = mid
    
    if tom < mid_1:
        left = mid + 1
    else:
        right = mid - 1

    #print(left,right)
    
print(best_k)
"""


"""

#Бинарный поиск
a = input()
b = input()

iron = int(a.split()[0])
wood = int(a.split()[1])

iron_p = int(b.split()[0])
wood_p = int(b.split()[1])

sp = []
new_p = iron_p
for i in range(iron):
    sp.append(new_p)
    new_p += iron_p

for i in range(wood):
    sp.append(new_p)
    new_p += wood_p

left = 0
right = len(sp)-1
value = (iron_p*iron+wood_p*wood)/2


while left <= right:
    mid = int((left+right)/2)
    
    if sp[mid] == value:
        print(mid+1)
        break
    elif sp[mid] < value:
        left = mid+1
    else:
        right = mid-1
    
   # print(left,right,mid, value)
    
"""

    

#Успеваемость
"""

a = int(input())
b = int(input())
c = int(input())

ch = a + b + c
ozenka = 2*a + 3*b + 4*c

fives_need = (3.5 * ch - ozenka) / 1.5

fives = 0 if fives_need <= 0 else round(fives_need)

print(fives)

a = int(input()) #2 
b = int(input()) #3
c = int(input()) #4

ch = a+b+c
ozenka = 2*a+3*b+4*c
cur_ozenka = round(ozenka/ch) 
fives = 0

while cur_ozenka < 4:
    fives += 1
    ozenka += 5
    cur_ozenka = round(ozenka/(fives+ch))

print(fives)

    int(ozenka/(fives+ch))+1 if ozenka/(fives+ch) == 0.5 else
"""
    
    
"""
l = int(input())
r = int(input())
a = int(input())

ch = 0

for i in range(1, r - l + 1):
    ch += i // a

print(ch)
"""

"""
l = int(input())
r = int(input())
a = int(input())

ch = 0
i = l

while i+a<=r and i<=l+a:
    q = a
    while i+q<=r:
        q = q+a
        ch += 1
    i += 1

print(ch)
"""

#НОК
"""
sp = input().split()
a,a1=int(sp[0]),int(sp[0])
b,b1=int(sp[1]),int(sp[1])
 
nod = 0
 
while True:
    maxim = max(a1,b1)
    minim = min(a1,b1)
    
    if maxim%minim == 0:
        nod=minim
        break
    else:
        if a1>b1: a1 = maxim%minim
        else: b1 = maxim%minim

#print(a1,a,b1,b, "НОК",int((a*b)/nod))
print(int(max(a,b)/min(a,b)) if a%b==0 else int(((a*b)/nod)/b))
"""

#Алгоритм Евклида
"""
sp = input().split()
a=int(sp[0])
b=int(sp[1])
 
nod = 0
 
while True:
    maxim = max(a,b)
    minim = min(a,b)
    
    if maxim%minim == 0:
        nod=minim
        break
    else:
        if a>b: a = maxim%minim
        else: b = maxim%minim
        
print(nod)
"""

#Делители
"""
a = int(input())
i=2

sp = []
while i*i <= a:
    if a%i==0:
        sp.append(i)    
        a = int(a/i)
    else:
        i += 1

if a > 1:
    sp.append(a)

print(' '.join(str(i) for i in sp) )
"""

#Число делителей
"""
a = int(input())
i = 1
deliteli=0

while i*i <= a:
    #print(i)
    if a % i == 0:
        deliteli += 1

        if i != a/i:
            deliteli += 1
    i += 1
    
print(deliteli)

"""

"""
a = int(input())
b = int(input())
k = int(input())

for i in range(a,b+1):
    summa = i*(i-1)//2+1
    #print(summa)
    
    kol_vo_chisel_v_stroke = min(k,i)
    
    stroka = ''
    
    for e in range(kol_vo_chisel_v_stroke):
        stroka += str(summa) + ' '
        summa += 1
        
    print(stroka)
""" 

#Комета Бармалея

"""
a = int(input())
b = int(input())
c = int(input())
 
i = a
 
if a%c!=0:i=a-(a%c)+c
 
print((b-i)//c+1)
"""

"""
a = int(input())
b = int(input())
c = int(input())

ch = 0
a=a-(a%c)+c*(a%c)

while a <= b:
    a += c
    ch+=1

print(ch)
"""


#Сумма
"""
a=input()
print(sum([int(i) for i in a.split()]))
"""

#Квадраты

"""
a = int(input())
b = int(input())

i = int(a**(1/2))
if i*i < a: i += 1
#print(i)

while i*i <= b:
    print(i*i)
    i += 1
"""

"""
a = int(input())
b = int(input())
k = int(input())

for i in range(a, b+1):
    s = 1 + (i-1)*i//2
    cnt = min(i, k)
    line = ' '.join(str(s + t) for t in range(cnt))
    print(line)
"""

#Лестница чисел
"""
a = int(input())
b = int(input())
k = int(input())
stroka = ''
last_num = a

for i in range(a,b+1,1):
    if i == 1: stroka += '1'
    else:
        last_iter_num = last_num
        for c in range(last_num+1, last_num+i+1):
            if c-last_iter_num <= k:
                stroka += str(c) + ' '
            last_num=c

    stroka += '\n'

for i in range(a, b+1):
    s = 1 + (i-1)*i//2
    cnt = min(i, k)
    line = ' '.join(str(s + t) for t in range(cnt))
    print(line)

print(stroka)        
"""

"""
for i in range(1,a+1,1):
    stroka = ''
    c = [stroka + f' {b}' for b in range(1,i+1,1)] 
    print(''.join(c))
"""

"""
from itertools import *


#Задание 5D6819, ответ: zyxw
def f(x,y,z,w):
    return ((x or (not y)) and not(y==z) and (not w))

for a1, a2, a3, a4 in product([0,1],repeat=4):
    table = [(1,0,a1,0),(1,a2,1,0),(a3,1,1,a4)]

    if len(table) == len(set(table)):
        for p in permutations('xyzw'):
            if [f(**dict(zip(p, r))) for r in table] == [1,1,1]:
                print(p)
"""

#Задание B58EBB, ответ: zxwy
"""
def f(w,x,y,z):
    return (((not(x<=z)) or (y<=w) or (not y)))

for a1,a2,a3,a4,a5,a6,a7 in product([0,1],repeat=7):
    table = [(1,0,a1,a2),(a3,1,0,a4),(0,a5,a6,a7)]

    if len(table) == len(set(table)):
        for p in permutations('wxyz'):
            if [f(**dict(zip(p, r))) for r in table] == [0,0,0]:
                print(p)
"""

#Задание 72DFBC, ответ: yxwz
"""
def f(x,y,z,w):
    return (((not x) or (not y)) and (not (y==z)) and (not w))

for a1, a2, a3, a4 in product([0,1],repeat=4):
    table = [(a1,0,0,1),(1,a2,0,a3),(0,1,a4,1)]

    if len(table) == len(set(table)):
        for p in permutations('xyzw'):
            if [f(**dict(zip(p, r))) for r in table] == [1,1,1]:
                print(p)
"""

#Задание 7848B2, ответ: xywz
"""
def f(x,y,z,w):
    return ((not x) or y or ((not z) and w))

table = [(1,0,0,0), (1,0,0,1), (1,0,1,1)]

for p in permutations('xyzw'):
    if [f(**dict(zip(p,r))) for r in table] == [0,0,0]:
        print(p)
"""

#Задание 75A1B7, ответ: zyxw
"""
def f(x,y,z,w):
    return ((x and (not y)) or (x==z) or w)

for a1, a2, a3, a4 in product([0,1],repeat=4):
    table = [(1,a1,0,0),(1,1,a2,0),(a3,1,1,a4)]

    if len(table) == len(set(table)):
        for p in permutations('xyzw'):
            if [f(**dict(zip(p, r))) for r in table] == [0,0,0]:
                print(p)
"""

#Задание 42A6BC, ответ: zyxw
"""
def f(x,y,z,w):
    return (((not x) or (not y)) and (not(y==z)) and w)

for a1, a2, a3, a4 in product([0,1],repeat=4):
    table = [(a1,0,0,1),(0,a2,0,a3),(1,0,a4,1)]

    if len(table) == len(set(table)):
        for p in permutations('xyzw'):
            if [f(**dict(zip(p, r))) for r in table] == [1,1,1]:
                print(p)
"""
#Задание 804673, ответ: zxyw
"""
def f(x,y,z,w):
    return ((not(x<=z))or(y==w) or y)

for a1, a2, a3, a4, a5, a6, a7 in product([0,1],repeat=7):
    table = [(1,0,a1,a2),(a3,1,0,a4),(0,a5,a6,a7)]

    if len(table) == len(set(table)):
        for p in permutations('xyzw'):
            if [f(**dict(zip(p, r))) for r in table] == [0,0,0]:
                print(p)   
"""

#Задание 3CF37C, ответ: xywz
"""
def f(x,y,z,w):
    return ((not(y<=w))or(x<=z)or(not x))

for a1, a2, a3, a4, a5, a6, a7 in product([0,1],repeat=7):
    table = [(a1,a2,0,0),(a3,1,a4,a5),(a6,0,1,a7)]

    if len(table) == len(set(table)):
        for p in permutations('xyzw'):
            if [f(**dict(zip(p, r))) for r in table] == [0,0,0]:
                print(p)    
"""

#Задание 6E287D, ответ: xywz
"""
def f(x,y,z,w):
    return (x and (not y) and ((not z) or w))

table = [(1,0,0,0), (1,0,1,0), (1,0,1,1)]

for p in permutations('xyzw'):
    if [f(**dict(zip(p,r))) for r in table] == [1,1,1]:
        print(p)
"""

#Задание C8147F, ответ: xzyw
"""
def f(x,y,z,w):
    return ((not(y<=z)or(x<=w)or(not x)))
 
for a1, a2, a3, a4, a5, a6, a7 in product([0, 1], repeat=7):
    table = [(a1, 0, a2, 0), (a3, a4, 1, a5), (a6, 1, 0, a7)]
    if len(table) == len(set(table)):
        for p in permutations('xyzw'):
            if [f(**dict(zip(p, r))) for r in table] == [0,0,0]:
                print(p)
"""