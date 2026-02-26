
from ipaddress import *

net = ip_network('17.234.25.1/255.255.224.0', 0)
print(str(net[-1]))