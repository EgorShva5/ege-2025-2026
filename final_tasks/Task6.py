from turtle import *

'''tracer(0)
screensize(5_000,5_000)
m = 5

for i in range(6):
    fd(71*m)
    rt(90)
    fd(73*m)
    rt(90)

up()

fd(18*m)
rt(90)
fd(22*m)
lt(90)

down()

for i in range(6):
    fd(45*m)
    rt(90)
    fd(58*m)
    rt(90)
cnt = 0

up()
for x in range(-100,100):
    for y in range(-100,100):
       goto(x*m,y*m)
       dot(2, 'red')

done()'''
'''tracer(0)
screensize(5_000, 5_000)
m = 5

begin_fill()
rt(45)
for i in range(3):
    rt(45)
    fd(10*m)
    rt(45)

rt(315)
fd(10*m)
rt(90)
fd(20*m)
rt(90)

for i in range(2):
    fd(10*m)
    rt(90)

end_fill()
up()

canvas = getcanvas()
cnt = 0
for x in range(-50,50):
    for y in range(-50, 50):
        if canvas.find_overlapping(x*m,y*m,x*m,y*m) == (5,):
            cnt += 1
        
        goto(x*m, y*m)


print(cnt)
done()'''

'''import tkinter as tk

win = tk.Tk()
cv1 = tk.Canvas(win, width=5000,height=500)
cv1.pack()
cv2 = tk.Canvas(win, width=5000, height=500)
cv2.pack()

t_1 = RawTurtle(cv1)
t_2 = RawTurtle(cv2)

t_1.speed(1000)
t_2.speed(1000)

t_1.up()
t_2.up()


m=1

t_1.down()
t_1.begin_fill()
t_1.pencolor('red')
for i in range(2):
    t_1.fd(32*m)
    t_1.rt(90)
    t_1.fd(38*m)
    t_1.rt(90)
    
    t_2.fd(32*m)
    t_2.rt(90)
    t_2.fd(38*m)
    t_2.rt(90)
    
t_1.end_fill()
t_1.up()
t_2.pencolor('blue')
t_2.fd(25*m)
t_2.rt(90)
t_2.fd(21*m)
t_2.lt(90)
t_2.begin_fill()
t_2.down()
for i in range(2):
    t_2.fd(29*m)
    t_2.rt(90)
    t_2.bk(18*m)
    t_2.rt(90)

t_2.end_fill()

cnt = 0
for x in range(-500,500):
    for y in range(-500,500):
        if (cv1.find_overlapping(x*m,y*m,x*m,y*m) != ()) and (cv2.find_overlapping(x*m,y*m,x*m,y*m)!=()):
            cnt+= 1
print(cnt)

done()'''

'''tracer(0)
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

done()'''