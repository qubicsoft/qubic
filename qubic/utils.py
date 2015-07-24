from __future__ import division, print_function

from progressbar import ProgressBar, Bar, ETA, Percentage
import numpy as np
import signal
import traceback

_NLEVELS = 0
_MAX_NLEVELS = 0


class _ProgressBar(ProgressBar):
    def __init__(self, poll=0.1, **keywords):
        global _NLEVELS, _MAX_NLEVELS
        self._nlevels = _NLEVELS
        _NLEVELS += 1
        _MAX_NLEVELS = max(_MAX_NLEVELS, _NLEVELS)
        ProgressBar.__init__(self, poll=poll, **keywords)
        if self._nlevels == 0:
            self._signal_old = signal.signal(signal.SIGINT, self._int_handler)
        print('\n\n\033[F', end='')

    def update(self, n=None):
        if n is not None:
            ProgressBar.update(self, n)
            return
        ProgressBar.update(self, self.currval + 1)
        if self.currval >= self.maxval:
            self.finish()

    def finish(self):
        global _NLEVELS, _MAX_NLEVELS
        ProgressBar.finish(self)
        if self._nlevels == 0:
            print((_MAX_NLEVELS - 1) * '\n', end='')
            signal.signal(signal.SIGINT, self._signal_old)
            _MAX_NLEVELS = 0
        else:
            print('\033[F\033[F', end='')
        _NLEVELS -= 1

    def _int_handler(self, signum, frame):
        global _NLEVELS, _MAX_NLEVELS
        _NLEVELS = 0
        _MAX_NLEVELS = 0
        signal.signal(signal.SIGINT, self._signal_old)
        e = KeyboardInterrupt()
        e.__traceback__ = traceback.extract_stack(frame)
        raise e


def progress_bar(n, info=''):
    """
    Return a default progress bar.

    Example
    -------
    >>> import time
    >>> n = 10
    >>> bar = progress_bar(n, 'LOOP')
    >>> for i in range(n):
    ...     time.sleep(1)
    ...     bar.update()

    """
    return _ProgressBar(widgets=[info, Percentage(), Bar('=', '[', ']'),
                                 ETA()], maxval=n).start()


def _compress_mask(mask):
    mask = mask.ravel()
    if len(mask) == 0:
        return ''
    output = ''
    old = mask[0]
    n = 1
    for new in mask[1:]:
        if new is not old:
            if n > 2:
                output += str(n)
            elif n == 2:
                output += '+' if old else '-'
            output += '+' if old else '-'
            n = 1
            old = new
        else:
            n += 1
    if n > 2:
        output += str(n)
    elif n == 2:
        output += '+' if old else '-'
    output += '+' if old else '-'
    return output


def _uncompress_mask(mask):
    i = 0
    l = []
    nmask = len(mask)
    while i < nmask:
        val = mask[i]
        if val == '+':
            l.append(True)
            i += 1
        elif val == '-':
            l.append(False)
            i += 1
        else:
            j = i + 1
            val = mask[j]
            while val not in ('+', '-'):
                j += 1
                val = mask[j]
            l.extend(int(mask[i:j]) * (True if val == '+' else False,))
            i = j + 1
    return np.array(l, bool)
