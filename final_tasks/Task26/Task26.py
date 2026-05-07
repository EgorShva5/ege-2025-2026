text = open('text.txt', encoding='UTF-8')
komp = int(text.readline())
polz = int(text.readline())

komps = [0]*komp
profit = [0]*komp

t_p = []
for i in range(polz):
    t_p.append(tuple(map(int, text.readline().split())))

t_p.sort()

cnt_cl = 0
for st,end in t_p:
    for i in range(komp):
        if komps[i] < st: 
            cnt_cl += 1
            komps[i] = end
            profit[i] += (end-st)*(end-st+1)//2
            break
        
print(cnt_cl, max(profit))            
            
    