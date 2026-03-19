from math import *
from turtle import *

data = [tuple(map(float, x.split())) for x in open('ClusterA.txt')]

def visualization(cluster):
    screensize(5000,5000)
    tracer(0)
    up()
    for cl, colour in (zip(cluster, ['red', 'green', 'blue'])):
        for x, y in cl:
            goto(x*10, y*10)
            dot(3, colour)
    done()
    
def get_centroid(cluster):
    r = []
    for p in cluster:
        r += [((sum(dist(p, p1) for p1 in cluster), p))]

    return min(r)[1]

clusters = []

while data:
    clusters.append([data.pop()])
    
    for p1 in clusters[-1]:
        neighbours = [a for a in data if dist(a, p1) < 1]
        clusters[-1].extend(neighbours)
        for b in neighbours: data.remove(b)


centroids = [get_centroid(cent) for cent in clusters]

visualization(clusters)

print(len(clusters[1]), len(clusters[0]), int(sum([dist(i, (1.0, 1.5)) for i in centroids])*10_000))
#print(int(100_000 * sum(p[0] for p in centroids)/2), int(100_000 * sum(p[1] for p in centroids)/2))