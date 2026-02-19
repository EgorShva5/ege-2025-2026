from ipaddress import *

net= ip_network('77.180.176.14/255.255.254.0', 0)

print(net[-2])