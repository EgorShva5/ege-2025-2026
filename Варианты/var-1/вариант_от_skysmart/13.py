from ipaddress import *

net = ip_network('210.102.240.138/255.255.255.224',0)

cnt = 0
for i in net:
    text = bin(int(i))[2:]
    
    if text.count('1') % 6 != 0:
        cnt += 1
        print(i)
print(cnt) 