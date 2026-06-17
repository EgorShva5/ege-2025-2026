text = open('text.txt', encoding='UTF-8').read()

k = 0
l = 0
ans = 0
for r in range(len(text)-1):
    if text[r] == 'D':
        k += 1
    
    while k > 1:
        if text[l] == 'D':
            k -= 1
        l += 1
    
    if k == 1:
        ans = max(ans, r-l+1)

print(ans)