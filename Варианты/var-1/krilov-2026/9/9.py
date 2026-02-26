text = open('text.txt', mode = 'r')

for e, i in enumerate(text):
    stroka = i.split()
    odni_chisla = list(map(int, [b for b in stroka if stroka.count(b) == 1]))
    tri_chisla = list(map(int, [b for b in stroka if stroka.count(b) == 3]))
    
    if len(odni_chisla) == 4 and len(tri_chisla) == 3 and sum(odni_chisla) > sum(tri_chisla):
        print(e+1)
    
    
    