from turtle import *

tracer(0)
screensize(5_000,5_000)
m = 10

down()
for i in range(5):
    dot(10,'red')
    up()
    goto(xcor()+0,ycor()+5*m)
    down()
    circle(5*m,180)
    
    up()
    goto(xcor()+5*m,ycor()+0*m)
    down()
    circle(5*m,180)

    up()
    goto(xcor()+0*m,ycor()-5*m)
    down()
    circle(5*m,180)
    
    up()
    goto(xcor()-5*m,ycor()+0*m)
    down()
    circle(5*m,180)

up()
for x in range(-50,50):
    for y in range(-50,50):
        goto(x*m,y*m)
        dot(2,'red')     

done()