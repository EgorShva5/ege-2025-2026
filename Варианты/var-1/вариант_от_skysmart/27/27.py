from math import hypot

def dist(p1,p2):
    return hypot(p1[0]-p2[0], p1[1]-p2[1])

def find_centroid(cluster):
    r = []
    for p1 in cluster:
        s = sum(dist(p1,p2) for p2 in cluster)
        r.append([s, p1])
    return min(r)[1]

def db_scan(filename):
    data = [tuple(map(float, i.split())) for i in open(f'{filename}.txt')]
    clusters = []
    
    while data:
        clusters.append([data.pop()])
        for i in clusters[-1]:
            neight = [b for b in data if dist(i,b) < 1]
            clusters[-1].extend(neight)
            for i in neight: data.remove(i)
        
    return clusters

def a():
    cl = db_scan('A')

    print(*map(len, cl))
    
    c1 = find_centroid(cl[0])
    c2 = find_centroid(cl[1])
    
    print(int((c1[0]+c2[0])/2*10_000), int((c1[1]+c2[1])/2*10_000))

def b():
    cl = db_scan('B')
    

    c1 = find_centroid(cl[0])
    c2 = find_centroid(cl[1])
    
    print(int((c1[0]+c2[0])/2*10_000), int((c1[1]+c2[1])/2*10_000))


a()
b()