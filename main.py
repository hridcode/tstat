import sys, csv, random, time
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

params = sys.argv[1:]

flags = set()
values = {}
to_read = {
    'X': 1,
    'y': 1,
    'c': 1,
    'm': 1,
    'j': 1,
    'p': 2
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

def type_or_throw(v, t, m=""):
    try:
        return t(v)
    except ValueError:
        throw(m or "bad data, check your input")

j = values.get('j')
if j == None:
    j = 6
else:
    j = type_or_throw(j[0], float, "invalid type provided for rounding place (j), check your input")
    if not j.is_integer():
        throw(f"rounding place (j) should be an integer, {int(j)} places will be used", 1)
    j = int(j)

def tofarray(s):
    try:
        return list(map(float, s.split(',')))
    except ValueError or TypeError:
        throw("non-numerical data presented, check your input")

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

def linear_regression(X, y):
    n = len(X)
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

    return m, b, r

def polynomial_regression(X, y, d):
    n = len(X)

    if n <= d:
        throw(f"not enough points to perform regression of degree {d}")

    sXs = [asum(X, y, i + 1, 0) for i in range(d * 2)][::-1] + [n,]
    sXys = [asum(X, y, i, 1) for i in range(d + 1)]

    sXs.reverse()

    coeff_matrix = np.array([
        sXs[i:i + d + 1][::-1] for i in range(d + 1)
    ])

    ans_matrix = np.array([
        [i, ] for i in sXys 
    ])

    coeffs = np.linalg.inv(coeff_matrix) @ ans_matrix
    return [i[0] for i in coeffs] 

def stdev(X):
    n = len(X)
    mean = sum(X) / n

    square_mean_diff = sum(map(lambda x: (x - mean) ** 2, X))

    population_variance = square_mean_diff / n
    sample_variance = square_mean_diff / (n - 1)

    population_stdev = population_variance ** 0.5
    sample_stdev = sample_variance ** 0.5

    return population_variance, sample_variance, population_stdev, sample_stdev

if mode == 'r':
    try:
        if csv_data:
            X, y = [float(i) for i in csv_data[values['X'][0]]], [float(i) for i in csv_data[values['y'][0]]]
        else:
            X, y = tofarray(values['X'][0]), tofarray(values['y'][0])
    except Exception:
        throw("error occurred while parsing data, check your input")

    n = len(X)

    match values.get('m', [''])[0]:
        case 'linear':
            m, b, r = linear_regression(X, y)

            print(Style.BRIGHT + 'Linear regression statistics:' + Style.RESET_ALL)
            print('\ny = ax + b\n')
            print(f'a = {round(m, j)}')
            print(f'b = {round(b, j)}\n')
            print(f'r = {round(r, j)}')
            print(f'R² = {round(r**2, j)}')
        case 'quadratic':
            a, b, c = polynomial_regression(X, y, 2)

            print(Style.BRIGHT + 'Quadratic regression statistics:' + Style.RESET_ALL)
            print('\ny = ax² + bx + c\n')
            print(f'a = {round(a, j)}')
            print(f'b = {round(b, j)}')
            print(f'c = {round(c, j)}')
        
            #print(f'R^2 = {r**2:.6f}')
        case 'cubic':
            a, b, c, d = polynomial_regression(X, y, 3)

            print(Style.BRIGHT + 'Cubic regression statistics:' + Style.RESET_ALL)
            print('\ny = ax³ + bx² + cx + d\n')
            print(f'a = {round(a, j)}')
            print(f'b = {round(b, j)}')
            print(f'c = {round(c, j)}')
            print(f'd = {round(d, j)}')
        case 'quartic':
            a, b, c, d, e = polynomial_regression(X, y, 4)

            print(Style.BRIGHT + 'Quartic (degree-4) regression statistics:' + Style.RESET_ALL)
            print('\ny = ax⁴ + bx³ + cx² + dx + e\n')
            print(f'a = {round(a, j)}')
            print(f'b = {round(b, j)}')
            print(f'c = {round(c, j)}')
            print(f'd = {round(d, j)}')
            print(f'e = {round(e, j)}')
        case 'quintic':
            a, b, c, d, e, f = polynomial_regression(X, y, 4)

            print(Style.BRIGHT + 'Quintic (degree-5) regression statistics:' + Style.RESET_ALL)
            print('\ny = ax⁵ + bx⁴ + cx³ + dx² + ex + f\n')
            print(f'a = {round(a, j)}')
            print(f'b = {round(b, j)}')
            print(f'c = {round(c, j)}')
            print(f'd = {round(d, j)}')
            print(f'e = {round(e, j)}')
            print(f'f = {round(f, j)}')
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

    population_variance, sample_variance, population_stdev, sample_stdev = stdev(X)

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
elif mode == 's':
    trials = type_or_throw(values['p'][0], int, "invalid parameter provided, check your input")
    samples = type_or_throw(values['p'][1], int, "invalid parameter provided, check your input")

    t1 = time.time_ns()

    results = []
    for t in range(trials):
        sample = []
        for x in range(samples):
            sample.append(random.randint(0, 1))
        results.append(sample.count(1) / samples)

    percentage = lambda p: round(p * 100, j)
    *_, standard_deviation, _ = stdev(results)

    t2 = time.time_ns()
    dt = (t2 - t1) / (10 ** 9)

    print(Style.BRIGHT + f"{trials}-trial simulation of {samples} coin flips:\n" + Style.RESET_ALL)
    print(f"Total of {samples * trials} flips, done in {dt} seconds\n")
    print(f"mean = {percentage(sum(results) / len(results))}%")
    print(f"median = {percentage(median(results))}%")
    print(f"σ = {percentage(standard_deviation)}%")
    print(f"min = {percentage(min(results))}% ({samples * min(results):.0f}/{samples} heads)")
    print(f"max = {percentage(max(results))}% ({samples * max(results):.0f}/{samples} heads)")