import sys, math
from colorama import Fore, Back, Style, init
import numpy as np

init()

def throw(message, mode = 0):
    if mode == 0:
        print(Fore.RED + Style.BRIGHT + 'error: ' + Style.RESET_ALL + message)
        sys.exit()
    else:
        print(Fore.YELLOW + Style.BRIGHT + 'warning: ' + Style.RESET_ALL + message)

"""
tstat -r -X [3,4,5] -y [4,5,6] -m linear
"""

params = sys.argv[1:]

flags = set()
values = {}
to_read = {
    'X': 1,
    'y': 1,
    'c': 1,
    'm': 1,
    'j': 1
}

read_flag = None

def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

for param in params:
    if param.startswith('-') and not isint(param):
        p = param.lstrip('-')
        flags.add(p)

        if p in to_read:
            read_flag = p
            values[p] = []
    else:
        if not read_flag: 
            throw("unexpected parameter " + param)
        values[read_flag].append(param)
        if len(values[read_flag]) == to_read[read_flag]:
            read_flag = None

modes = {'r', 'd', 'n', 's'}
if len(modes & flags) == 0:
    throw("no mode selected")
elif len(modes & flags) > 1:
    throw("more than one mode selected, please pick only one")

mode = (modes & flags).pop() 

tofarray = lambda s: list(map(float, s.split(',')))
def asum(ax, ay, dx, dy):
    return sum([(ax[i] ** dx) * (ay[i] ** dy) for i in range(len(ax))])

if mode == 'r':
    X, y = tofarray(values['X'][0]), tofarray(values['y'][0])
    n = len(X)

    j = values.get('j')
    if j == None:
        j = 6
    else:
        j = int(j[0])

    match values.get('m', ['linear'])[0]:
        case 'linear':
            sX, sy = sum(X), sum(y)
            sX2 = asum(X, y, 2, 0)
            sy2 = asum(X, y, 0, 2)
            sXy = asum(X, y, 1, 1)
            m = ((n * sXy) - (sX * sy)) / ((n * sX2) - (sX ** 2))
            b = (sy - m * sX) / n

            r = ((n * sXy) - (sX * sy)) / (
                ((n * sX2) - (sX ** 2)) *
                ((n * sy2) - (sy ** 2))
            ) ** 0.5

            print(Style.BRIGHT + 'Linear regression statistics:' + Style.RESET_ALL)
            print('\ny = ax + b\n')
            print(f'a = {round(m, j)}')
            print(f'b = {round(b, j)}\n')
            print(f'r = {round(r, j)}')
            print(f'R^2 = {round(r**2, j)}')
        case 'quadratic':
            sX, sy = sum(X), sum(y)
            sX2, sX3, sX4 = asum(X, y, 2, 0), asum(X, y, 3, 0), asum(X, y, 4, 0)
            sXy, sX2y = asum(X, y, 1, 1), asum(X, y, 2, 1)

            coeff_matrix = np.array([
                [sX2, sX, n],
                [sX3, sX2, sX],
                [sX4, sX3, sX2]
            ])

            ans_matrix = np.array([
                [sy],
                [sXy],
                [sX2y]
            ])

            a, b, c = np.linalg.inv(coeff_matrix) @ ans_matrix
            a, b, c = a[0], b[0], c[0]
            
            print(Style.BRIGHT + 'Quadratic regression statistics:' + Style.RESET_ALL)
            print('\ny = ax^2 + bx + c\n')
            print(f'a = {round(a, j)}')
            print(f'b = {round(b, j)}')
            print(f'c = {round(c, j)}')
                        
            #print(f'R^2 = {r**2:.6f}')