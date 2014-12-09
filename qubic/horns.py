from __future__ import division
from pysimulators import LayoutGrid

__all__ = []


class HornLayout(LayoutGrid):
    def plot(self, edgecolor_open='black', edgecolor_closed='black',
             facecolor_open='white', facecolor_closed='0.2', **keywords):
        LayoutGrid.plot(self[self.open], edgecolor=edgecolor_open,
                        facecolor=facecolor_open, **keywords)
        LayoutGrid.plot(self[~self.open], edgecolor=edgecolor_closed,
                        facecolor=facecolor_closed, **keywords)
