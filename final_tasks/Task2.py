from itertools import *

'''print('x y z w')
for x in 0,1:
    for y in 0,1:
        for z in 0,1:
            for w in 0,1:
                if x and not y and (not z or w):
                    print(x,y,z,w)'''

'''def f(x,y,z,w):
    return (x and not y) or (x==z) or not w

for a1,a2,a3,a4 in product([0,1],repeat=4):
    table = [(0,1,1,0),(0,a1,a2,a3),(a4,1,0,1)]
    
    if len(table) == len(set(table)):
        for p in permutations('xyzw'):
            if [f(**dict(zip(p,r))) for r in table] == [0,0,0]:
                print(p)'''

'''def f(x,y,z,w):
    return (x or y) and not (y==z) and not w

for a1,a2,a3,a4 in product([0,1], repeat=4):
    table = [(1,a1,1,a2),(0,1,a3,0),(a4,1,1,0)]
    
    if len(table) == len(set(table)):
        for p in permutations('xyzw'):
            if [f(**dict(zip(p,r))) for r in table] == [1,1,1]:
                print(p)
'''
'''print('x y z w')
for x in 0, 1:
  for y in 0, 1:
    for z in 0, 1:
      for w in 0, 1:
        F = (x and not y and (not z or w))
        if F:
          print(x, y, z, w)
          
def f(a,b,c,d):
    return (a and not b and (not c or d))

table = [(0,0,1,0),(0,0,1,1),(1,0,1,1)]

if len(table) == len(set(table)):
    for p in permutations('abcd'):
        if [f(**dict(zip(p,r))) for r in table] == [1,1,1]:
            print(p)'''

'''from itertools import *

def f(a,b,c,d):
    return ((a and not d) or (d==c) or not b)

for a1,a2,a3,a4 in product([0,1], repeat=4):
    table = [(a1,1,0,a2),(0,1,0,1),(0,a3,a4,0)]
    
    if len(table) == len(set(table)):
        for p in permutations('abcd'):
            if [f(**dict(zip(p,r))) for r in table] == [0,0,0]:
                print(p)'''

#Пробник от ЦУ
'''from itertools import *

def f(x,y,z,w):
    return (not z and (y or not w) or (z and w) or not x)

for a1,a2,a3,a4,a5 in product([0,1], repeat=5):
    table = [(a1,0,1,0), (0,0,1,a2), (a3,a4,a5,1)]
    
    if len(table) == len(set(table)):
        for p in permutations('xyzw'):
            if [f(**dict(zip(p,r))) for r in table] == [0,0,0]:
                print(p)
            '''

#F=(w≡z)∨¬(y→w)∨¬x
'''from itertools import *

def f(x,y,w,z):
    return (w==z) or not (y <= w) or not x

for a1,a2,a3,a4,a5 in product([0,1], repeat=5):
    table = [(a1,0,1,0),(a2,1,1,a3),(0,a4,a5,0)]
    
    if len(table) == len(set(table)):
        for p in permutations('xyzw'):
            if [f(**dict(zip(p,r))) for r in table] == [0,0,0]:
                print(p)'''

'''from itertools import *

#	№ 25341 ЕГКР 13.12.25 (Уровень: Базовый)
#F=(w≡z)∨¬(y→w)∨¬x 
#Ответ: zwxy

def f(x,y,w,z):
    return ((w==z) or (not (y<=w)) or (not x))

for a1,a2,a3,a4,a5 in product([0,1], repeat=5):
    table = [(a1,0,1,0),(a2,1,1,a3),(0,a4,a5,0)]
    
    if len(table) == len(set(table)):
        for p in permutations('xywz'):
            if [f(**dict(zip(p,r))) for r in table] == [0,0,0]:
                print(p)
'''
'''
def f(x,y,z,w):
    return ((not(y<=w))or(x<=z)or(not x))

for a1, a2, a3, a4, a5, a6, a7 in product([0,1],repeat=7):
    table = [(a1,a2,0,0),(a3,1,a4,a5),(a6,0,1,a7)]

    if len(table) == len(set(table)):
        for p in permutations('xyzw'):
            if [f(**dict(zip(p, r))) for r in table] == [0,0,0]:
                print(p)
'''