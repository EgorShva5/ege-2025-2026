import re
a=re.findall('[1-9A-D][0-9A-D]*[02468AC]',open('text.txt').readline())
print(max(len(x) for x in a)) 