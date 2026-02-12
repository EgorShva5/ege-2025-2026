from itertools import *

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
def f(x,y,z,w):
    return ((not(y<=w))or(x<=z)or(not x))

for a1, a2, a3, a4, a5, a6, a7 in product([0,1],repeat=7):
    table = [(a1,a2,0,0),(a3,1,a4,a5),(a6,0,1,a7)]

    if len(table) == len(set(table)):
        for p in permutations('xyzw'):
            if [f(**dict(zip(p, r))) for r in table] == [0,0,0]:
                print(p)
'''