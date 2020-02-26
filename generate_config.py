import  matplotlib.pyplot as plt 
import matplotlib.patches as pat
import cv2
import json
import time

def finish():
    global rects, borders
    print(rects)
    print(borders)
    data = {}
    data['xY1'] = rects[0] / 100
    data['xY3'] = rects[2] / 100
    data['xY4'] = rects[3] / 100
    data['xY2'] = rects[1] / 100
    data['danger1'] = borders[0]
    data['danger2'] = borders[1]

    with open('config.json', 'w') as outfile:
        json.dump(data, outfile, indent=3)
    outfile.close()

    plt.close()

def onClick(event):
    global count, rects, borders
    # print(f'button={event.button}, x={event.x}, y={event.y} xdata={event.xdata}, ydata={event.ydata}')
    plt.scatter(event.xdata, event.ydata, 10)
    plt.plot(event.xdata, event.ydata, ',')
    if count == 6:
        finish()

    if count < 4:
        rects.append([int(event.xdata), int(event.ydata)])  
        if len(rects) > 1 and len(rects) != 4:
            x = rects[count - 1]
            y = rects[count]
            plt.plot([x[0], y[0]], [x[1], y[1]], 'r-')
        else:
            w = rects[count - 1]
            x = rects[count]
            y = rects[0]
            plt.plot([w[0], x[0]], [w[1], x[1]], 'r-')
            plt.plot([x[0], y[0]], [x[1], y[1]], 'r-')
    else:
        borders.append([int(event.xdata), int(event.ydata)])
        if len(borders) == 2:
            x = borders[0]
            y = borders[1]
            # plt.plot([x[0], y[0]], [x[1], y[1]], 'g-')
            rect = pat.Rectangle(x, y[0]-x[0], y[1]-x[1], fill=False)
            plt.gca().add_patch(rect)

    count += 1
    fig.canvas.draw()
    
(W, H) = [int(x) for x in input('Enter Width and Height: ').split()]
path = input('Enter image path: ')
fig = plt.figure()

rects = []
borders = []
img = cv2.imread(path)
resized = cv2.resize(img, (W, H))
plt.imshow(resized)
count = 0

cid = fig.canvas.mpl_connect('button_press_event', onClick)
plt.show()
