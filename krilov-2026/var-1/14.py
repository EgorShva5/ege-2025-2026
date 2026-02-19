a = 3*2187**1801 + 729**2000 - 4*243**2100 + 81**2200 - 2*27**2400-13122

def to_27(number):
    chisla = '0123456789abcdefghijklmnopqrstuvwxyz'
    
    result = ''
    while number > 0:
        result = chisla[number % 27] + result
        number //= 27
    
    return result

ts_a = to_27(a)
cnt = 0
chisla = '9abcdefghijklmnopqrstuvwxyz'
for d in ts_a:
    if d in chisla: cnt += 1
    
print(cnt)