#Пробный ЕГЭ от ЦУ
def f(n):
    bin_n = bin(n)[2:]
    print(bin_n)
    if len(bin_n) % 2 == 0:
        bin_n = bin_n + bin_n[len(bin_n)//2-1:len(bin_n)//2+1]
    else:
        bin_n = bin_n + bin_n[len(bin_n)//2-1:len(bin_n)//2+2]
    
    return int(bin_n,2)

for i in range(1,50):
    res = f(i)
    
    if res > 145:
        print(res)
        break