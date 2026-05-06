from math import dist

def find_centroid(cluster):
    r = []
    for i in cluster:
        summa = sum(dist(i,i1) for i1 in cluster)
        r.append([summa, i])
    
    print(r)
    return min(r)[1]

def db_scan(filename):
    clusters = []
    
    cur_data = [tuple(map(float, i.split())) for i in open(f'{filename}.txt')]
    
    while cur_data:
        clusters.append([cur_data.pop()])
        
        for i in clusters[-1]:
            close_by = [a for a in cur_data if dist(a,i) < 1]
            clusters[-1].extend(close_by)
            for b in close_by: cur_data.remove(b)
    return clusters

def a():
    clusters = db_scan('ClusterA')
    c_1 = find_centroid(clusters[0])
    c_2 = find_centroid(clusters[1])
    
    cnt = 0
    for i in clusters[1]:
        if i[0] > c_2[0]: cnt += 1
    
    print(cnt, abs(c_1[0]-c_2[0])*10_000) 

def b():
    clusters = db_scan('ClusterB')
    clusters.sort(key=len)
    
    c_b, c_m, c_s = find_centroid(clusters[-1]),find_centroid(clusters[-2]),find_centroid(clusters[0])

    cnt = 0    
    for i in clusters[0]: 
        if abs(i[0] - c_s[0]) < 0.9 and abs(i[1] - c_s[1]) < 0.9:
            cnt += 1
    print(cnt)
b()