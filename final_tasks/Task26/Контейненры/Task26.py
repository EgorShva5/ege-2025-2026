text = open('text.txt')

s, n = map(int, text.readline().split())
masses = sorted([int(i) for i in text])

cur_mas = 0
cnt = 0
for i in masses:
    if cur_mas + i > s:
        break
    
    cur_mas += i
    cnt += 1

print(n-cnt, sum(masses) - cur_mas)
