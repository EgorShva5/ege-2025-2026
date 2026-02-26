from turtle import *

mult = 4
screensize(5000,5000)
tracer(0)


down()

for i in range(2):
    for b in range(2):
        fd(180*mult)
        rt(120)
    rt(120)

rt(150)
fd(15*mult)
rt(90)
fd(360*mult)
rt(90)
fd(15*mult)
rt(30)
fd(74*mult)

up()
for x in range(-50,50):
    for y in range(-50,50):
        goto(x*mult,y*mult)
        dot(3, 'red')

update() 
done()
