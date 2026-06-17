from itertools import *

for e,i in enumerate(product('ОРШЭЯ', repeat=7)):
    print(e, i)
    if e == 30:
        print(e,i)
        break