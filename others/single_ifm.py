import sys
import math

import numpy as np

def str_to_float(x) :
    a = x.split('&');
    a = [x.split('/') for x in a];
    g_l, l_l = [], [];
    for v in a :
        g, l = v;
        g_l.append(float(g));
        l_l.append(float(l));
    
    return g_l, l_l;


def l_to_ifm(ga, la) :
    return 100 * np.abs(ga - la) / (ga + la);

def print_g_ifm(x) :
    ga, la = str_to_float(x);
    ga, la = np.array(ga), np.array(la);

    ifma = l_to_ifm(ga, la);
    
    ga = np.append(ga, ga.mean());
    ifma = np.append(ifma, ifma.mean());

    print(np.round(ga, 1));
    print(np.round(la, 1));
    print(np.round(ifma, 1));    

    pstr = '';
    for g, ifm in zip(ga, ifma) :
        pstr += ' & ';
        pstr += str(round(g, 1));
        pstr += ' & ';
        pstr += str(round(ifm, 1));

    pstr += ' \\\\';

    print(pstr);

x = '81.4/70.8 & 62.6/90.5 & 52.9/96.3 & 46.3/97.5 & 34.8/99.4 & 31.5/97.9'
x = '81.7/84.9 & 72.2/69.3 & 65.9/81.7 & 57.3/77.2 & 51.7/71.8 & 42.7/71.4'
x = '86.5/74.5 & 82.2/68.2 & 74.7/72.9 & 72.6/76.4 & 65.5/55.0 & 62.3/60.2'

x = '66.4/91.8 & 52.3/86.5 & 35.0/95.9 & 24.5/97.2 & 20.5/96.8 & 15.1/95.8'
x = '65.7/93.4 & 58.1/86.1 & 49.5/94.1 & 45.1/94.2 & 42.7/93.3 & 39.5/93.0'
x = '73.0/80.0 & 66.9/77.1 & 60.8/83.8 & 57.9/80.5 & 52.1/81.8 & 49.0/86.3'

x = '72.0/79.7 & 61.5/71.2 & 53.2/64.5 & 50.7/56.1 & 46.6/55.7 & 39.9/39.4'
x = '64.8/77.3 & 51.5/70.7 & 43.5/78.3 & 41.2/75.7 & 36.4/78.7 & 32.3/75.5'

x = '68.3/76.8 & 56.0/71.1 & 49.0/70.4 & 44.4/66.7 & 39.9/73.1 & 36.8/69.2'

print_g_ifm(x);
