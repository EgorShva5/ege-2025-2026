text = open('text.txt', encoding='UTF8').readlines()

max_sum = 0
cnt = 0
for i in range(len(text)-1):
    a,b = int(text[i]), int(text[i+1])
    
    if (a % 5 == 0 and b % 5 != 0) or (a % 5 != 0 and b % 5 == 0) :
        cnt += 1
        max_sum = max(max_sum, a+b)

print(cnt, max_sum)