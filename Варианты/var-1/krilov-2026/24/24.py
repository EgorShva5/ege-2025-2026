from re import findall

f = open('text.txt', mode='r', encoding='UTF-8').readline()

a = findall(r'[S]')