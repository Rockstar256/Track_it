import numpy as np
import matplotlib.pyplot as plt


def draw_parabola(x, y):
    if len(x) == 0:
        return
    p1 = np.polyfit(x, y, 2)
    x_new = np.linspace(x[0], x[-1], 50)
    z_new = np.polyval(p1, x_new)
    plt.plot(x, y, 'o', color='red', label='Points')
    plt.plot(x_new, z_new, '-', color='blue', label='Curve')
    plt.legend()
    plt.show()


x_list = []
y_list = []
with open('track.txt', mode='rt', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        if line == 'stop':
            draw_parabola(np.array(x_list), np.array(y_list))
            x_list = []
            y_list = []
        else:
            x_, y_ = tuple(line.split(' '))
            x_list.append(int(x_))
            y_list.append(int(y_))



