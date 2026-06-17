text = open('text.txt', encoding='UTF-8').read()

k = 0
l = 0
ans = 0
for r in range(len(text)-1):
    if text[r] == 'C' and text[r+1] == 'D':
        k += 1
    
    while k > 140:
        if text[l] == 'C' and text[l+1] == 'D':
            k -= 1
        l += 1
    
    if k == 140:
        ans = max(ans, r-l+2)

print(ans)