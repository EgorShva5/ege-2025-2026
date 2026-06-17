from turtle import *

tracer(0)
screensize(5_000,5_000)
m = 5

for i in range(2):
    fd(6*m)
    rt(90)
    fd(9*m)
    rt(90)

up()

fd(1*m)
rt(90)
fd(3*m)
lt(90)

down()

for i in range(2):
    fd(7*m)
    rt(90)
    fd(9*m)
    rt(90)

up()

for x in range(-50,50):
    for y in range(-50,50):
        goto(x*m,y*m)
        dot(2,'red')

done()