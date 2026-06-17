s = open('text.txt').readline()

for c in 'EIOUY': s = s.replace(c,'A')

c = ''
m = 10000
for r in range(len(s)):
    c+=s[r]
    if c[-1]=='A':
        while c.count('20')>=26:
            if c.count('20')==26:
                m=min(m,len(c))
            c = c[1:]
        c = ''
print(m)