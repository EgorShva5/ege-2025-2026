'''text = open('text.txt', encoding = 'UTF-8')

n,m,k = map(int, text.readline().split())'''

'''7 7 8
1 1
6 6
5 5
6 7
4 4
2 2
3 3'''

with open("text.txt") as f:
    n, rows, sits_per_row = map(int, f.readline().split())
    sits = [list(map(int, l.split())) for l in f]
    matrix = [[True] * sits_per_row for _ in range(rows)]

for row, sit in sits:
    matrix[row - 1][sit - 1] = False

print(matrix)
def check_pair(i):
    c = 0
    for row in matrix:
        sit1 = row[i - 1]
        sit2 = row[i]
        if sit1 and sit2:
            c += 1
        else:
            return c


maxrow, minsit = 0, 0
for i in range(1, len(matrix[0])):
    r = check_pair(i)
    if r > maxrow:
        maxrow = r
        minsit = i
print(maxrow, minsit)