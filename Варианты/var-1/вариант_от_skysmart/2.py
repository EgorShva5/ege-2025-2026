from itertools import permutations, product

def f(x,y,z,w):
    return (not (z == (not y)) or (x == (y or not w)))

for a1,a2,a3,a4 in product([0,1],repeat=4):
    table = [(1,a1,0,1),(a2,1,1,0),(1,0,a3,a4)]
    
    if len(table) == len(set(table)):
        for p in permutations('xywz'):
            if [f(**dict(zip(p,r))) for r in table] == [0,0,0]:
                print(p)
    