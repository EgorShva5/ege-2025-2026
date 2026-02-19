from turtle import *

tracer(0)
screensize(5000,5000)

m = 5

down()
for i in range(4):
    fd(9*m)
    lt(180)
    backward(10*m)
    rt(90)

up()
backward(7*m)
lt(90)
fd(3*m)
rt(90)

down()
for i in range(2):
    fd(17*m)
    lt(90)
    fd(20*m)
    lt(90)

up()
for x in range(-50,50):
    for y in range(-50,50):
        goto(x*m, y*m)
        dot(2,'red')

done()
update()
    
    