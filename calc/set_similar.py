# -*- coding: utf-8 -*-

def jaccard(x, y):
    x = frozenset(x)
    y = frozenset(y)
    return len(x & y) / float(len(x | y))

def dice(x, y):
    x = frozenset(x)
    y = frozenset(y)
    return 2 * len(x & y) / float(sum(map(len, (x, y))))

def simpson(x, y):
    x = frozenset(x)
    y = frozenset(y)
    return len(x & y) / float(min(map(len, (x, y))))
