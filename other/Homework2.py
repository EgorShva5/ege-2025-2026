#57DBBA, 98
"""
file = open('text9.txt','r',encoding='UTF-8')

ch = 0

for i in file:
    a = sorted([int(b) for b in i.split()])
    
    b = [ c for c in a if a.count(c) == 2 ]
    c = [ q for q in a if a.count(q) == 1 ]
    
    #try:
    #    print(b[0],b[2],a[-1], len(set(b)) == 2 and len(c) == 3 and b[0] != a[-1] and b[2] != a[-1])
    #except:
    #    pass
    #print(len(set(b)),set(b), len(c))
    
    if len(set(b)) == 2 and len(c) == 3 and b[0] != a[-1] and b[2] != a[-1]:
        ch += 1

print(ch)"""

#B42173, 2640
"""file = open('text8.txt','r',encoding='UTF-8')

ch = 0

for i in file:
    a = sorted([int(b) for b in i.split()])
    
    if (a[0]+a[-1])**2 > (a[1]**2 + a[2]**2 +a[3]**2):
        ch += 1

print(ch)"""

#40F07C, 245
"""file = open('text3.txt','r',encoding='UTF-8')

ch = 0

for i in file:
    a = sorted([int(b) for b in i.split()])
    
    b = [ c for c in a if a.count(c) == 3 ]
    c = [ q for q in a if a.count(q) == 1 ]
    
    
    #print(len(set(b)),set(b), len(c))
        
    if len(set(b)) == 1 and len(c) == 3:
        if (3*(b[1]**2))>(c[0]**2+c[1]**2+c[2]**2):
            ch += 1

print(ch)
"""

#E20F0A, 7695
"""file = open('text6.txt','r',encoding='UTF-8')

ch = 0

for i in file:
    a = sorted([int(b) for b in i.split()])
    
    if len(set(a))==5 and (a[-1]+a[0])*3 >= (a[1]+a[2]+a[3])*2:
        ch += 1

print(ch)
"""

#9F8A0F, 83
"""file = open('text5.txt','r',encoding='UTF-8')

ch = 0

for i in file:
    a = sorted([int(b) for b in i.split()])
    
    b = [ c for c in a if a.count(c) == 2 ]
    c = [ q for q in a if a.count(q) == 1 ]
    
    
    #print(len(set(b)),set(b), len(c))
    
    if len(set(b)) == 2 and len(c) == 3:
        if ((sum(b))/4)<(sum(c)/3):
            ch += 1

print(ch)"""

#82E1FA, 116
"""file = open('text4.txt','r',encoding='UTF-8')

ch = 0

for i in file:
    a = sorted([int(b) for b in i.split()])
    
    if a[-1] < sum(a)-a[-1] and (a[0] + a[1] == a[2] + a[3] or a[0] + a[3] == a[1] + a[2] or a[1] +a[3] == a[0] + a[2]):
        ch += 1

print(ch)"""

#6DD746, 273
"""
file = open('text3.txt','r',encoding='UTF-8')

ch = 0

for i in file:
    a = sorted([int(b) for b in i.split()])
    
    b = [ c for c in a if a.count(c) == 3 ]
    c = [ q for q in a if a.count(q) == 1 ]
    
    
    #print(len(set(b)),set(b), len(c))
        
    if len(set(b)) == 1 and len(c) == 3:
        if (sum(b)**2)>(sum(c)**2):
            ch += 1

print(ch)
"""

#90D64F, 96
"""
file = open('text2.txt','r',encoding='UTF-8')

ch = 0

for i in file:
    a = sorted([int(b) for b in i.split()])
    
    b = [ c for c in a if a.count(c) == 2 ]
    c = [ q for q in a if a.count(q) == 1 ]
    
    
    #print(len(set(b)),set(b), len(c))
    
    if len(set(b)) == 2 and len(c) == 3:
        if ((sum(b))/4)<(sum(c)/3):
            ch += 1

print(ch)
"""

#B9BE47, 2623
"""
file = open('text1.txt','r',encoding='UTF-8')

ch = 0

for i in file:
    a = sorted([int(b) for b in i.split()])
    
    if (a[0]+a[-1])**2 > (a[1]**2 + a[2]**2 +a[3]**2):
        ch += 1

print(ch)
"""