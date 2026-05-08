

'''
def get_primes(n):
    a = [True] * (n+1)
    a[0] = False
    a[1] = False
    
    for i in range(2, n):
        if a[i]:
            for j in range(i*i, n+1, i):
                a[j] = False
    
    final_list = [b for b in range(1,n+1) if a[b]]
    return final_list

#89_428_304
primes = get_primes(100_000_000)

def check_n(n):
    list_of_dels = []
    first_n = n
    i = 0
    while True:
        d = primes[i]
        
        if d > n: break
        
        if n % d == 0:
            list_of_dels.append(d)
            n //= d
        else:
            i += 1
    
    if len(list_of_dels) >= 6 and first_n % sum(list_of_dels) == 0:
        return list_of_dels
    else: return 0
    
c = 0
chislo = 89_428_305
while c < 6:
    check = check_n(chislo)
    
    if check:
        c += 1
        print(chislo, sum(check))

    chislo += 1
'''
'''
def get_primes(n):
    spis = [True] * (n+1)
    spis[0] = False
    spis[1] = False
    
    for i in range(2, n+1):
        if spis[i]:
            for j in range(i*i,n+1,i):
                spis[j] = False
    
    final_list = [b for b in range(2,n+1) if spis[b]]
    return final_list

primes = get_primes(6_000_000)
#3 502 100
def check_n(n):
    dels_cnt = 0
    last = 0
    polindrom_found = False
    i = 0
    
    while True:
        delit = primes[i]
        
        if delit > n: break
        
        if n % delit == 0:
            last = delit
            n //= delit
            dels_cnt += 1
            if len(str(delit)) == 2 and str(delit)[0] == str(delit)[1]:
                polindrom_found = True
        else:
            i += 1
        
    if dels_cnt == 4 and polindrom_found:
        return last
    else: 
        return 0 

c = 0
f_ch = 3_502_100
while c < 5:
    tr = check_n(f_ch)
    if tr:
        c += 1
        print(f_ch, tr)
    f_ch += 1
'''
    
'''
cnt = 0
cur_ch = 8_996_452

def find_dels(ch):
    dels = []
    ch_2 = ch
    i = 2
    while ch_2 > 1:
        if ch_2 % i == 0:
            ch_2 //= i
            dels.append(i)
        else:
            i += 1
    return dels

while cnt < 5:
    cur_ch += 1
    cur_dels = find_dels(cur_ch)
    
    print(cur_ch)
    if len(cur_dels) == 2:
        th_th = [i for i in cur_dels if str(i).count('3') == 2]
        if len(th_th) == 2:
            print(cur_ch, max(cur_dels))
            cnt += 1
'''

'''from re import fullmatch

for i in range(0, 10**10, 9874):
    if fullmatch('89.*6.7.9.', str(i)):
        print(i, i//9874)'''
'''
27778 

Назовём маской числа последовательность цифр, в которой также могут встречаться следующие символы:
– символ «?» означает ровно одну произвольную цифру;
– символ «*» означает любую последовательность цифр произвольной длины; в том числе «*» может задавать и пустую последовательность.
Например, маске 123*4?5 соответствуют числа 123405 и 12300405.
Среди натуральных чисел, не превышающих 108, найдите все числа, соответствующие маске 12??15*6, делящиеся на 271 без остатка.
В ответе запишите в первом столбце таблицы все найденные числа в порядке возрастания, а во втором столбце – 
соответствующие им результаты деления этих чисел на 271.
Количество строк в таблице для ответа избыточно.
'''

'''
from re import compile

r = compile('^12\d\d15\d*6$')

for i in range(0,10**8, 271):
    if r.match(str(i)):
        print(i, i//271)

'''