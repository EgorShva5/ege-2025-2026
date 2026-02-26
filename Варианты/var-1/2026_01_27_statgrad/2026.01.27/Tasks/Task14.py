nums = sorted('0123456789qwertyuiopasdfghjklzxcvbnm')

for x in nums[:22]:
    x1 = '2496' + str(x) + '2'
    x2 = '8' + str(x) + '223'
    x3 = '2331768' + str(x) + '3'
    
    res = int(x1, 21) + int(x2, 21) + int(x3, 21)
    
    if res % 20 == 0:
        res = res // 20
        print(res)
        break