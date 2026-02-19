from itertools import *

for e,i in enumerate(product('АЕЛРСТ', repeat = 5)):
    if (e+1) % 2 == 0:
        i = ''.join(i)
        if i[0] != 'А' and i[0] != 'С' and i[0] != 'Т' and 'ЛЛ' not in i and i.count('Л') == 2:
            print(e+1, i)
