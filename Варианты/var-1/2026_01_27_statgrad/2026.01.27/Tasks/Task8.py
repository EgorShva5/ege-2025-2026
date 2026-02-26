from itertools import *

for e, i in enumerate(product('АДЛОСФЦЩ', repeat = 4)):
    if e %2 == 0:
        if i[0] != 'А' and i[-1] != 'А' and i.count('Л') >= 3:
            print(i, e+1)
            break
    