from itertools import *

def f(x,y,z,w):
    return (x <= (not ((y <= z) and (z == (not w)))))

for a1,a2,a3,a4,a5 in product([0,1], repeat=5):
    table = [(0,a1,a2,0), (a3,1,1,a4), (a5,1,0,0)]
    
    if len(set(table)) == len(table):
        for p in permutations('xyzw'):
            if [f(**dict(zip(p,r))) for r in table] == [0,0,0]:
                print(p)