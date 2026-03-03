'''def to_3(n):
    all_nums = '0123456789'
    
    final_num = ''
    while n > 0:
        final_num = all_nums[n%3] + final_num
        n //= 3
        
    return final_num

def sum_count(n):
    summa = 0
    n = int(n)
    while int(n) > 0:
        summa += n%10
        n //= 10
    return summa

def f(n):
    new_n = to_3(int(n))
    
    if int(n) % 3 == 0:
        new_n = new_n + new_n[-2:]
    else:
        new_n = new_n + to_3(sum_count(new_n)*3)
    
    return int(new_n,3)

res = 0
dt = 1000
for n in range(1, 1000):
    r = f(n)
    delta = abs(826 - r)
    if delta < dt:
        dt = delta
        res = r
        
print(res)
'''