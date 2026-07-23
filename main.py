import sys, csv
from colorama import Fore, Back, Style, init
import numpy as np
from collections import Counter

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

def median(a):
    a.sort()
    n = len(a)
    if n % 2 == 1:
        return a[int(n // 2)]
    else:
        middle = int(n // 2)
        return (a[middle] + a[middle - 1]) / 2

csv_route = values.get('c', [None])[0]
csv_data = {}

if csv_route:
    with open(csv_route, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            if not csv_data.keys():
                for key in row:
                    csv_data[key] = []

            for key in row:
                csv_data[key].append(row[key])

j = values.get('j')
if j == None:
    j = 6
else:
    j = int(j[0])

if mode == 'r':
    if csv_data:
        X, y = [float(i) for i in csv_data[values['X'][0]]], [float(i) for i in csv_data[values['y'][0]]]
    else:
        X, y = tofarray(values['X'][0]), tofarray(values['y'][0])

    n = len(X)

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
            print(f'R² = {round(r**2, j)}')
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
            print('\ny = ax² + bx + c\n')
            print(f'a = {round(a, j)}')
            print(f'b = {round(b, j)}')
            print(f'c = {round(c, j)}')
                        
            #print(f'R^2 = {r**2:.6f}')
        case _:
            throw("unsupported regression mode")
elif mode == "d":
    if csv_data:
        X = [float(i) for i in csv_data[values['X'][0]]]
    else:
        X = tofarray(values['X'][0])

    X.sort()

    n = len(X)
    mean = sum(X) / n
    med = median(X)
    mod = None

    hashmap = Counter(X)
    hashmap = dict(sorted(hashmap.items(), reverse=True))
    count_values = hashmap.values()

    if min(count_values) != max(count_values):
        mod = list(filter(lambda x: hashmap[x] == max(count_values), hashmap.keys()))

    range = max(X) - min(X)

    q1s, q3s = [], []
    middle = int(n // 2)
    if n % 2 == 1:
        q1s = X[:middle]
        q3s = X[middle+1:]
    else:
        q1s = X[:middle]
        q3s = X[middle:]

    q1 = median(q1s)
    q3 = median(q3s)

    iqr = q3 - q1

    square_mean_diff = sum(map(lambda x: (x - mean) ** 2, X))

    population_variance = square_mean_diff / n
    sample_variance = square_mean_diff / (n - 1)

    population_stdev = population_variance ** 0.5
    sample_stdev = sample_variance ** 0.5

    print(Style.BRIGHT + 'Dataset statistics:' + Style.RESET_ALL)
    print(f'\nn = {n}\n')
    print(f'mean = {round(mean, j)}')
    print(f'median = {round(med, j)}')
    print(f'mode = {", ".join(map(str, mod)) if mod else 'N/A'}\n')
    print(f'range = {round(range, j)}')
    print(f'min = {round(min(X), j)}')
    print(f'max = {round(max(X), j)}\n')
    print(f'Q₁ = {round(q1, j)}')
    print(f'Q₃ = {round(q3, j)}')
    print(f'IQR = {round(iqr, j)}\n')
    print(f's = {round(sample_stdev, j)}')
    print(f'σ = {round(population_stdev, j)}')
    print(f's² = {round(sample_variance, j)}')
    print(f'σ² = {round(population_variance, j)}\n')
    print(f'Σx = {round(sum(X), j)}')
    print(f'Σx² = {round(sum([i ** 2 for i in X]), j)}')
elif mode == 'n':
    pass