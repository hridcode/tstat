import sys
import math
from colorama import Fore, Back, Style

flags_single = ['X', 'y', 'd', 'r', 'h', 'j', 'm', 'c']
flags_double = ['dataset', 'regression', 'round', 'mode', 'csv', 'help']

def isnumber(s):
    return s.replace('.', '').isdecimal()

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
            return flag
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
            current_values.append(float(arg) if isnumber(arg) else arg)

    if current_values:
        flags[current_flag] = tuple(current_values)

    return flags

flags = parse_flags()

round_decimal = flags.get('j', (6,))[0]