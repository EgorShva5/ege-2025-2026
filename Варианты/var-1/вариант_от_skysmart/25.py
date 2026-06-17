from re import compile

r = compile('^243\d{2}5\d*1$')

cnt = 0
for i in range(0,10**8+1, 127):
    if r.match(str(i)):
       cnt += 1
       print(i, i//127)
       
    if cnt == 2: break 