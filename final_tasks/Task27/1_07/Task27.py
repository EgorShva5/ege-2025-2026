from math import dist

def find_centroid(cluster):
    r = []
    for p1 in cluster:
        summa = sum(dist(p1,p2) for p2 in cluster)
        r.append([summa, p1])
    return min(r)[1]

def db_scan(filename: str):
    data = [tuple(map(float, i.split())) for i in open(f'{filename}.txt')]
    clusters = []
    
    while data:
        clusters.append([data.pop()])
        for i in clusters[-1]:
            close_by = [a for a in data if dist(a,i) < 1]
            clusters[-1].extend(close_by)
            for b in close_by: data.remove(b)
    
    return clusters

def a():
    clusters = db_scan('ClusterA')
    clusters.sort(key=len)
    
    c_s = find_centroid(clusters[0])
    c_b = find_centroid(clusters[-1])
    
    distance = dist(c_s, [1.0, 1.5]) + dist(c_b, [1.0, 1.5])
    print(max(map(len, clusters)), int(distance*10_000))

def b():
    clusters = db_scan('ClusterB')
    clusters.sort(key=len)
    
    c_m = find_centroid(clusters[1])
    c_b = find_centroid(clusters[-1])
    
    cnt = 0
    for i in clusters[1]:
        if dist(i, c_m) < 1.2:
            cnt += 1
    
    min = float('inf')
    for i in clusters[-1]:
        min_dist = dist(i, c_b)
        if min_dist != 0.0 and min > min_dist:
            min = min_dist
            
    print(cnt-1, int(min*10_000))
    
b()
#a()
#print(*map(len,db_scan('ClusterA')))