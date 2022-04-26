import collections
from collections import Set


class OrderedSet(Set):
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, item):
        return item in self.d

    def __iter__(self):
        return iter(self.d)
