s = open('text.txt').readline()
l = 0
k = 0
ans = 0
for r in range(len(s) - 1):
    if s[r] == 'B' and s[r + 1] == 'C':
        k += 1
    while k > 190:
        if s[l] == 'B' and s[l + 1] == 'C':
            k -= 1
        l += 1
    if k == 190:
        ans = max(ans, r - l + 2)
print(ans)