#29074 
from math import hypot
from re import match

def dist(p1,p2):
    return hypot(p1[0]-p2[0], p1[1]-p2[1])

def find_centroid(cluster):
    r = []
    for p1 in cluster:
        summa = sum(dist(p2,p1) for p2 in cluster)
        r.append([summa,p1])
    return min(r)[1]

def db_scan(filename):
    data = [(float(i.split()[0]), float(i.split()[1]), i.split()[-1]) for i in open(f'{filename}.txt')]
    clusters = []
    
    while data:
        clusters.append([data.pop()])
        for i in clusters[-1]:
            neighbours = [a for a in data if dist(i, a) < 1]
            clusters[-1].extend(neighbours)
            for b in neighbours: data.remove(b)
    return clusters

def a():
    clusters = db_scan('ClusterA')
    clusters.sort(key=len)
    
    f_cl_cnt = 0
    for i in clusters[0]:
        if i[2][0] == 'Z':
            f_cl_cnt += 1
            
    s_cl_cnt = 0
    for b in clusters[-1]:
        if b[2][0] == 'Z':
            s_cl_cnt += 1
    
    sm_cent = find_centroid(clusters[0])
    big_cent = find_centroid(clusters[-1])
    
    print(min(f_cl_cnt, s_cl_cnt), max(f_cl_cnt, s_cl_cnt))

def b():
    clusters = db_scan('ClusterB')
    clusters.sort(key=len)
    
    centr = [find_centroid(clusters[0]), find_centroid(clusters[1]),find_centroid(clusters[-1])]

    maxim = 0
    minim = float('inf')
    for e,i in enumerate(clusters):
        for a in i:
            if match(r'^L[0-9]*V$', a[2]):
                distance = dist(a, centr[e])
                minim = min(minim, distance)
                maxim = max(maxim, distance)
    print(int(minim*10_000), int(maxim*10_000))
    
a(), b()
 #   print(*map(len, db_scan('ClusterA')))