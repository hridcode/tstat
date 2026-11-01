import sys
import math
import numpy as np
from colorama import Fore, Back, Style

# -----------------------
# flag helper functions
# -----------------------

flags_single = ['X', 'y', 'd', 'r', 'h', 'j', 'm', 'c']
flags_double = [None, None, 'dataset', 'regression', 'help', 'round', 'regmode', 'csv']

def isnumber(s):
    return s.replace('.', '').isdecimal()

def strarray(s):
    if ',' not in s:
        return
    try:
        return [float(i) for i in s.split(',')]
    except ValueError:
        return

def convert_value(arg):
    arg_array = strarray(arg)
    arg_number = float(arg) if isnumber(arg) else arg

    if arg_array: return arg_array
    if arg_number: return arg_number
    return arg

def get_flag(arg):
    arg = arg.strip()
    if not arg[0] == '-':
        return 

    double_flag = False

    if arg[1] == '-':
        double_flag = True

    flag = arg[2:] if double_flag else arg[1:]

    if double_flag:
        if flag in flags_double:
            return flags_single[flags_double.index(flag)]
        else:
            return
    else:
        if flag in flags_single:
            return flag
        else:
            return

def parse_flags():
    flags = {}
    
    current_flag = None
    current_values = []

    for arg in sys.argv:
        if get_flag(arg):
            if current_flag: flags[current_flag] = tuple(current_values)    
            current_values = []
            current_flag = get_flag(arg)
        else:
            current_values.append(convert_value(arg))

    if current_values:
        flags[current_flag] = tuple(current_values)

    return flags

# ------------------------
# flag parsing + variables
# ------------------------

flags = parse_flags()
flag_keys = list(flags.keys())

decj = int(flags.get('j', (6,))[0])

mode = ''
if 'd' in flag_keys:
    mode = 'd'
elif 'r' in flag_keys:
    mode = 'r'

if not mode:
    pass

data_X, data_y = flags.get('X', [[],])[0], flags.get('y', [[],])[0]

# --------------------
# regression functions
# --------------------

def log_array(a):
    return [math.log(i) for i in a]

def degree_sum(X, y, dX, dy):
    return sum([X[i]**dX * y[i]**dy for i in range(len(X))])

def r2_score(y, y_pred):
    y_mean = sum(y) / len(y)
    squared_residuals = [(y_pred[i] - y[i]) ** 2 for i in range(len(y))]
    total_square_sum = [(y[i] - y_mean) ** 2 for i in range(len(y))]
    return 1 - (sum(squared_residuals) / sum(total_square_sum))

def linear_regression(X, y):
    n = len(X)
    sX, sy = degree_sum(X, y, 1, 0), degree_sum(X, y, 0, 1)
    sX2, sy2 = degree_sum(X, y, 2, 0), degree_sum(X, y, 0, 2)
    sXy = degree_sum(X, y, 1, 1)

    m = (sy * sX2 - sX * sXy) / (n * sX2 - sX ** 2)
    b = (n * sXy - sX * sy) / (n * sX2 - sX ** 2)

    r = (n * sXy - sX * sy) / (
        (n * sX2 - sX ** 2) * (n * sy2 - sy ** 2)
    ) ** 0.5

    return {
        'coefficients': (m, b),
        'correlation': r,
        'r2_score': r ** 2
    }

def eval_polynomial(X, *coeffs):
    result = 0
    degree = len(coeffs) - 1
    for index, coeff in enumerate(coeffs):
        result += coeff * X ** (degree - index)
    return result

def polynomial_regression(X, y, d):
    A = np.matrix([
        [degree_sum(X, y, i + j, 0) if i + j > 0 else len(X) for j in range(d + 1)][::-1]
        for i in range(d + 1)
    ])
    B = np.matrix([
        [degree_sum(X, y, i, 1)] for i in range(d + 1)
    ])

    sol = np.linalg.inv(A) @ B
    coefficients = tuple([i[0] for i in sol.tolist()])
    y_pred = [eval_polynomial(i, *coefficients) for i in X]

    return {
        'coefficients': coefficients,
        'r2_score': r2_score(y, y_pred)
    }

# --------------------
# cli outputs
# --------------------

if mode == 'r':
    if len(data_X) != len(data_y):
        pass

    regression_modes = ['linear', 'growth', 'exponential', 'logarithmic', 'power', 'quadratic', 'cubic', 'quartic', 'quintic']
    regression_mode_selected = flags.get('m', ('linear',))[0]

    if regression_mode_selected not in regression_modes:
        pass

    print()
    match regression_mode_selected:
        case "linear":
            data = linear_regression(data_X, data_y)

            print(f"{Style.BRIGHT}{Fore.BLUE}Linear regression statistics:{Style.RESET_ALL}")
            print()
            print(f"{Fore.GREEN}y = ax + b{Style.RESET_ALL}\n")
            print(f"a = {data['coefficients'][0]:.{decj}f}")
            print(f"b = {data['coefficients'][1]:.{decj}f}")
            print()
            print(f"r = {data['correlation']:.{decj}f}")
            print(f"R² = {data['r2_score']:.{decj}f}")
        case "power":
            a, b = linear_regression(log_array(data_X), log_array(data_y))['coefficients']
            a = math.exp(a)
            
            r2 = r2_score(data_y, [
                a * (i ** b) for i in data_X
            ])

            print(f"{Style.BRIGHT}{Fore.BLUE}Power regression statistics:{Style.RESET_ALL}")
            print()
            print(f"{Fore.GREEN}y = ax^b{Style.RESET_ALL}\n")
            print(f"a = {a:.{decj}f}")
            print(f"b = {b:.{decj}f}")
            print()
            print(f"R² = {r2:.{decj}f}")
        case "growth":
            a, b = linear_regression(data_X, log_array(data_y))['coefficients']
            a, b = math.exp(a), math.exp(b)
            
            r2 = r2_score(data_y, [
                a * (b ** i) for i in data_X
            ])

            print(f"{Style.BRIGHT}{Fore.BLUE}Growth regression statistics:{Style.RESET_ALL}")
            print()
            print(f"{Fore.GREEN}y = ab^x{Style.RESET_ALL}\n")
            print(f"a = {a:.{decj}f}")
            print(f"b = {b:.{decj}f}")
            print()
            print(f"R² = {r2:.{decj}f}")
        case "exponential":
            a, b = linear_regression(data_X, log_array(data_y))['coefficients']
            a = math.exp(a)
            
            r2 = r2_score(data_y, [
                a * math.e ** (b * i) for i in data_X
            ])

            print(f"{Style.BRIGHT}{Fore.BLUE}Power regression statistics:{Style.RESET_ALL}")
            print()
            print(f"{Fore.GREEN}y = a * e^bx{Style.RESET_ALL}\n")
            print(f"a = {a:.{decj}f}")
            print(f"b = {b:.{decj}f}")
            print()
            print(f"R² = {r2:.{decj}f}")
        case "logarithmic":
            a, b = linear_regression(log_array(data_X), data_y)['coefficients']
            
            r2 = r2_score(data_y, [
                a + b * math.log(i) for i in data_X
            ])

            print(f"{Style.BRIGHT}{Fore.BLUE}Logarithmic regression statistics:{Style.RESET_ALL}")
            print()
            print(f"{Fore.GREEN}y = a + b * ln(x){Style.RESET_ALL}\n")
            print(f"a = {a:.{decj}f}")
            print(f"b = {b:.{decj}f}")
            print()
            print(f"R² = {r2:.{decj}f}")
        case "quadratic":
            data = polynomial_regression(data_X, data_y, 2)
            a, b, c = data['coefficients']
            r2 = data['r2_score']

            print(f"{Style.BRIGHT}{Fore.BLUE}Quadratic regression statistics:{Style.RESET_ALL}")
            print()
            print(f"{Fore.GREEN}y = a + b * ln(x){Style.RESET_ALL}\n")
            print(f"a = {a:.{decj}f}")
            print(f"b = {b:.{decj}f}")
            print(f"c = {c:.{decj}f}")
            print()
            print(f"R² = {r2:.{decj}f}")
    print()
else:
    pass