# coding: U8
# file taken from https://github.com/michurin/pprofiler

from __future__ import print_function

import collections
import functools
import time
import math


__all__ = ['profiler']  # the only publick symbol
__version__ = '2.0'


SUBSCOPE_NAME = '~'


Scope = collections.namedtuple('Scope', ('stat', 'scopes'))


class Stat(object):

    def __init__(self):
        self.sum = self.sum2 = 0.
        self.min = self.max = None
        self.n = 0

    def update(self, val):
        if self.n == 0:
            self.min = self.max = float(val)
        else:
            self.min = min(self.min, val)
            self.max = max(self.max, val)
        self.sum += val
        self.sum2 += val * val
        self.n += 1

    @property
    def stat(self):
        avg = dev = None
        if self.n > 0:
            avg = self.sum / self.n
        if self.n > 1:
            dev = math.sqrt(
                (self.sum2 - self.sum * self.sum / self.n) /
                (self.n - 1)
            )
        return {
            'sum': self.sum,
            'num': self.n,
            'avg': avg,
            'dev': dev,
            'min': self.min,
            'max': self.max,
        }

    def __repr__(self):
        return '<{}({})>'.format(type(self).__name__, ', '.join('{}={!r}'.format(*kv) for kv in self.stat.items()))


class Timer(object):

    def __init__(self, scopes, name):
        self.scopes = scopes
        self.name = name
        self.start = None

    def __call__(self, f):
        @functools.wraps(f)
        def pprofiler_wrapper(*a, **kv):
            with self:
                return f(*a, **kv)
        return pprofiler_wrapper

    def __enter__(self):
        self.start = time.time()
        self.scopes._enter(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scopes._update(time.time() - self.start)
        return False


class TableField(object):

    def __init__(self, name, align_left=None, fill=None, extra_padding=None):
        self.name = name
        self.width = len(name)
        self.align = '<' if align_left else '>'
        self.fill = '.' if fill else ''
        self.extra_padding = 1 if extra_padding is None else extra_padding

    def update_width(self, width):
        self.width = max(self.width, width + self.extra_padding)

    def format_header(self):
        return '{:{}{}s}'.format(self.name, self.align, self.width)

    def format_separator(self):
        return '-' * self.width

    def format_value(self, value):
        return '{:{}{}{}s}'.format(value, self.fill, self.align, self.width)


class Profiler(object):

    def __init__(self):
        self.stack = [Scope(stat=None, scopes={})]

    def __call__(self, name):
        return Timer(self, name)

    def _enter(self, name):
        self.stack.append(self.stack[-1].scopes.setdefault(name, Scope(stat=Stat(), scopes={})))

    def _update(self, val):
        self.stack.pop().stat.update(val)

    @property
    def report(self):
        return scopes_to_report(self.stack[0].scopes)

    @property
    def is_complete(self):
        return len(self.stack) == 1

    def check_complete(self):
        if not self.is_complete:
            raise RuntimeError('pprofiler: report can not be prepared, not all measurements completed')

    def __iter__(self):
        return report_to_flat(self.report)

    @property
    def lines(self):
        report_fields = [
            TableField('name', align_left=True, fill=True, extra_padding=2),
            TableField('perc'),
            TableField('sum'),
            TableField('n'),
            TableField('avg'),
            TableField('max'),
            TableField('min'),
            TableField('dev'),
        ]
        lines = []
        for s in self:
            d = {k: '-' if s[k] is None else '{:.2f}'.format(s[k]) for k in ('sum', 'avg', 'min', 'max', 'dev')}
            d['name'] = '. ' * s['level'] + s['name'] + ' '
            d['perc'] = '{:.0f}%'.format(s['percent'])
            d['n'] = '{:d}'.format(s['num'])
            lines.append(d)
        fields = {f.name: f for f in report_fields}
        for l in lines:
            for n, f in l.items():
                fields[n].update_width(len(f))
        yield ' '.join(f.format_header() for f in report_fields)
        yield ' '.join(f.format_separator() for f in report_fields)
        for l in lines:
            yield ' '.join(f.format_value(l[f.name]) for f in report_fields)

    def print_report(self, printer=None):
        if printer is None:
            printer = print
        for s in self.lines:
            printer(s)


def report_to_flat(nodes):
    stack = []
    while stack or nodes:
        n = nodes.pop(0)
        n['level'] = len(stack)
        subscope = n.pop(SUBSCOPE_NAME, None)
        yield n
        if subscope:
            stack.append(nodes)
            nodes = subscope
        while len(nodes) == 0 and len(stack) > 0:
            nodes = stack.pop()


def scopes_to_report(scopes):
    r = []
    a = 0.
    for k, v in scopes.items():
        s = v.stat.stat
        s['name'] = k
        if v.scopes:
            sub_scopes = scopes_to_report(v.scopes)
            if len(sub_scopes) > 0:
                s[SUBSCOPE_NAME] = sub_scopes
        r.append(s)
        a += s['sum']
    if a > 0:
        p = lambda x: 100 * x['sum'] / a
    else:
        p = lambda x: 0.
    for i in r:
        i['percent'] = p(i)
    r = [x for x in r if x['num'] > 0 or len(x.get(SUBSCOPE_NAME, [])) > 0]
    r.sort(key=lambda x: x['sum'], reverse=True)
    return r


profiler = Profiler()
