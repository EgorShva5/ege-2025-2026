'''
qs = {
    (' ', 0): (' ', 1, 1),
    (' ', 1): (' ', 2, 1),
    
    ('0', 1): ('1', 1, 1),
    
    ('1', 1): ('0',1,1)
}

def f(s):
    s = list(' '+s+' ')
    i = 0
    q = 0
    
    while True:
        cmd = qs[(s[i], q)]
        
        s[i] = cmd[0]
        
        if cmd[1] == 2: break
        
        i += cmd[1]
        q = cmd[2]
        
    return ''.join(s)

for x in range(1, 70_000):
    if int(f(bin(x)[2:]),2) == 415:
        print(x)
'''

'''qs = {
    (' ', 0): ('1', 1, 1),
    (' ', 1): ('1', 1, 2),
    (' ', 2): ('1', 1, 3),
    (' ', 3): ('0', 2, 3),
    
    ('0', 1): ('1', 1, 1),
    
    ('1', 1): ('0', 1, 1)   
}

def f(s):
    s = list(' ' + s + '   ')
    q = 0
    i = 0
    
    while True:
        cmd = qs[(s[i],q)]
       # print(cmd, s[1])
        
        s[i] = cmd[0]
        
        if cmd[1] == 2: break
        
        i += cmd[1]
        q = cmd[2]
    
    return ''.join(s)

for x in range(1,10_000):
    if str(f(bin(x)[2:])) == str(bin(11_438)[2:]):
        print(x)
        break'''
    
