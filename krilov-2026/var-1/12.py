p = {
    (' ', 0): (' ', -1, 1), 
    (' ', 1): (' ', 2, 1),
    
    ('1', 1): ('0', 2, 1),
    
    ('0', 1): ('1', -1, 1)
}

def f(s):
    s = list(' ' + s + ' ')
    q = 0
    i = -1
    
    while True:
        cmd = p[(s[i], q)]
        
        s[i] = cmd[0]
        
        if cmd[1] == 2: break
        
        i += cmd[1]
        q = cmd[2]
    return ''.join(s)

for x in range(1,1000):
    res = f(bin(x)[2:])
    
    if res.count('0') == 125:
        print(bin(x)[2:])