from itertools import *

text = open('task.txt', mode='r', encoding='UTF-8')

for i in text:
    i = i.split()

    trizdi = [int(b) for b in i if i.count(b) == 3]
    dvazdi = [int(b) for b in i if i.count(b) == 2]
    edinozdi = [int(b) for b in i if i.count(b) == 1]
    
    if len(trizdi) == 3 and len(dvazdi) == 2 and len(edinozdi) == 2:
        if sum(edinozdi) <= min(min(dvazdi), min(trizdi)):
            print(i)
    