# FGBuster
# Copyright (C) 2019 Davide Poletti, Josquin Errard and the FGBuster developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import numpy as np


__all__ = [
    'MixingMatrix',
]


class MixingMatrix(tuple):
    """ Collection of Components
    The goal is to provide ways to evaluate all the components (or their
    derivatives) with a single call and store them in a matrix (the mixing
    matrix).
    There are two ways:
    - evaluate it using (nu, param_0, param_1, param_2, ...)
    - provide A_ev, which takes a single array as argument
    """
    def __new__(cls, *components):
        return tuple.__new__(cls, components)

    def __init__(self, *components):
        self.__first_param_of_comp = []
        self.__comp_of_param = []
        for i_c, c in enumerate(components):
            self.__first_param_of_comp.append(self.n_param)
            self.__comp_of_param += [i_c] * c.n_param

    @property
    def params(self):
        # TODO: handle components with the same name
        return ['%s.%s' % (type(c).__name__, p)
                for c in self for p in c.params]

    @property
    def defaults(self):
        return [d for c in self for d in c.defaults]

    @property
    def components(self):
        return [type(c).__name__ for c in self]

    @property
    def n_param(self):
        return len(self.__comp_of_param)

    @property
    def comp_of_dB(self):
        return self.__comp_of_param

    def eval(self, nu, *params):
        if params:
            shape = np.broadcast(*params).shape + (len(nu), len(self))
        else:
            shape = (len(nu), len(self))
        res = np.zeros(shape)
        for i_c, c in enumerate(self):
            i_fp = self.__first_param_of_comp[i_c]
            res[..., i_c] += c.eval(nu, *params[i_fp: i_fp + c.n_param])
        return res

    def evaluator(self, nu, unpack=(lambda x: x.reshape((-1,)))):
        if self.n_param:
            def f(param_array):
                param_array = np.array(param_array)
                return self.eval(nu, *[p for p in unpack(param_array)])
        else:
            A = self.eval(nu)
            def f():
                return A
        return f

    def diff(self, nu, *params):
        if not params:
            return None
        res = []
        for i_c, c in enumerate(self):
            param_slice = slice(self.__first_param_of_comp[i_c],
                                self.__first_param_of_comp[i_c] + c.n_param)
            res += [g[..., np.newaxis]
                    for g in c.diff(nu, *params[param_slice])]
        return res

    def diff_evaluator(self, nu, unpack=(lambda x: x.reshape((-1,)))):
        if self.n_param:
            def f(param_array):
                param_array = np.array(param_array)
                return self.diff(nu, *[p for p in unpack(param_array)])
        else:
            return None
        return f

    def diff_diff(self, nu, *params):
        if not params:
            return None
        res = [[np.zeros((1,1))
                for i in range(self.n_param)] for i in range(self.n_param)]
        for i_c, c in enumerate(self):
            param_slice = slice(self.__first_param_of_comp[i_c],
                                self.__first_param_of_comp[i_c] + c.n_param)
            comp_diff_diff = c.diff_diff(nu, *params[param_slice])
            i_start = param_slice.start
            for i in range(i_start, param_slice.stop):
                for j in range(param_slice.start, param_slice.stop):
                    res[i][j] = (
                        comp_diff_diff[i - i_start][j - i_start].reshape(-1, 1))
        return res
    
    def diff_diff_evaluator(self, nu, unpack=(lambda x: x.reshape((-1,)))):
        def f(param_array):
            param_array = np.array(param_array)
            return self.diff_diff(nu, *[p for p in unpack(param_array)])
        return f