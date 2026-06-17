text = open('text.txt', encoding='UTF-8').readlines()

def cnt_of_dels(n):
    dels = set()
    for i in range(2,int(n**0.5)+1):
        if n % i == 0: 
            dels.add(i)
            if i != n // i: dels.add(n//i)
    return dels
    
f_cnt = 0
for i in text:
    i.strip('\n')
    
    for b in i.split():
        print(b, cnt_of_dels(int(b)))
        if len(cnt_of_dels(int(b))) > 4:
            f_cnt += 1

print(f_cnt)