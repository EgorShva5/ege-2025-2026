p = {
    (' ', 0): ('1', 1, 1), 
    (' ', 1): ('0', 1, 2),
    (' ', 2): ('0', 1, 3),
    (' ', 3): ('1', 2, 3),
    
    ('0', 1): ('0', 1, 1),
    
    ('1', 1): ('1', 1, 1)
}

def f(s):
    s = list(' ' + s + '    ')
    q = 0
    i = 0
    
    while True:
        #print(s[i], i, q)
        cmd = p[(s[i], q)]

        s[i] = cmd[0]
        
        if cmd[1] == 2: break
        
        i += cmd[1]
        q = cmd[2]
    return ''.join(s)

print(int(f(bin(145_682)[2:]),2))