from itertools import *

qs = {
    (' ', 0): (' ', -1, 1),
    (' ', 1): (' ', 2, 1),
    
    ('1',1): ('0',2,1),
    
    ('0',1): ('1',-1,1)
}

def f(s):
    s = list(' '+s+' ')
    
    q = 0
    i = -1
    
    while True:
        data = qs[(s[i], q)]
        
        s[i] = data[0]
        
        if data[1] == 2: break
        
        q = data[2]
        i += data[1]
    
    return ''.join(s)

for e,i in enumerate(product('01', repeat=1000)):
    print(''.join(i))
    if f(''.join(i)).count('0') == 605:
        print(i.count('0'))