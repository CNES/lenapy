# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
import operator
import numbers


@xr.register_dataset_accessor("xharmo")
class HarmoSet:
    """
    This class implement an extension of any dataset to add some methods related to spherical harmonics decomposition
    The initial dataset must contain the necessary fields to define the spherical harmonics properties
    """

    def __init__(self, xarray_obj):
        """
        Initialise the HarmoSet object and test its integrity

        Parameters
        ----------
        xarray_obj : xr.Dataset
            Input dataset
        """
        self._obj = xarray_obj
        if 'l' not in xarray_obj.coords:
            raise AssertionError("The degree coordinates that should be named 'l' does not exist")
        if 'm' not in xarray_obj.coords:
            raise AssertionError("The order coordinates that should be named 'm' does not exist")
        if 'clm' not in xarray_obj.keys():
            raise AssertionError("The Dataset have to contain 'clm' variable")
        if 'slm' not in xarray_obj.keys():
            raise AssertionError("The Dataset have to contain 'slm' variable")

    def __neg__(self):
        self._obj.clm = -self._obj.clm
        self._obj.slm = -self._obj.slm
        return self

    def __abs__(self):
        self._obj.clm = abs(self._obj.clm)
        self._obj.slm = abs(self._obj.slm)
        return self

    def __iter__(self, dim='time'):
        """
        Iterate over one dimension that is 'time' by default
        Return a xr.Dataset at each iteration

        Parameters
        ----------
        dim : str
            Dimension of the xarray over which iterate

        """
        if dim in self._obj.coords:
            # TODO see add .xharmo ?
            return iter(self._obj.groupby(dim))
        else:
            return [self]

    def __add__(self, other):
        return self._apply_operator(operator.add, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._apply_operator(operator.sub, other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        return self._apply_operator(operator.mul, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._apply_operator(operator.truediv, other)

    def __rtruediv__(self, other):
        self._obj.clm = 1/self._obj.clm
        self._obj.slm = 1/self._obj.slm
        return self.__mul__(other)

    def __pow__(self, power):
        if isinstance(power, numbers.Number):
            return self._apply_operator(operator.pow, power)
        else:
            raise AssertionError("Cannot raise to power with an object ", power)

    def _apply_operator(self, op, other):
        """
        Generic function to overwrite the operator of HarmoSet object
        Apply the operator to clm and slm variables

        Parameters
        ----------
        op : operator. Operation
            function from operator library to apply
        other : int, float, complex, xr.Dataset or HarmoSet
            second variable for the operation (self being the first variable)

        Returns
        -------
        HarmoSet
            return the object after the operation (that is also self apply)

        Raises
        ------
        AssertionError
            This function cannot operate on a HarmoSet with time dimension to self without time dimension.
        """
        # case where other is a number (int, float, complex)
        if isinstance(other, numbers.Number):
            self._obj.clm = op(self._obj.clm, other)
            self._obj.slm = op(self._obj.slm, other)

        # case where other is another xr.DataSet correspond to an HarmoSet
        else:
            if not isinstance(other, HarmoSet):
                # TODO try catch error and add verbose if needed
                other = other.xharmo

            # change clm and slm size if other.l or other.m is smaller
            new_l = min(self._obj.l.size, other._obj.l.size)
            new_m = min(self._obj.m.size, other._obj.m.size)
            if self._obj.l.size != new_l or self._obj.m.size != new_m:
                self._obj.clm = self._obj.clm[:new_l, :new_m]
                self._obj.slm = self._obj.slm[:new_l, :new_m]

            # case where other does not have a time dimension
            if 'time' not in other._obj.coords:
                # TODO test if error in multi-dimension ensemble
                self._obj.clm = op(self._obj.clm, other._obj.clm[:new_l, :new_m])
                self._obj.slm = op(self._obj.slm, other._obj.slm[:new_l, :new_m])

            elif 'time' not in self._obj.coords:  # if the previous test, other has time dimension
                raise AssertionError("Cannot operate on a HarmoSet with time dimension to a Harmoset without it. "
                                     "Inverse the order of the HarmoSet in the operation.")

            # case where both xr.Dataset have a time dimension
            else:
                pass
                # TODO deal with missig month
                # old_month = self.month
                # exclude1 = set(self.month) - set(temp.month)
                # self.month = np.array(list(sorted(set(self.month) - exclude1)))
                # self.time = np.array([self.time[i] for i in range(len(self.time)) if not (old_month[i] in exclude1)])
                # for i in range(len(old_month)):
                #     for j in range(len(temp.month)):
                #         if old_month[i] == temp.month[j]:
                #             self.clm[:l1, :m1, i] += temp.clm[:l1, :m1, j]
                #             self.slm[:l1, :m1, i] += temp.slm[:l1, :m1, j]

                # to_keep = []
                # for i in range(len(old_month)):
                #     if not (old_month[i] in exclude1):
                #         to_keep.append(i)
                # self._obj.clm = self._obj.clm[:, :, to_keep]
                # self._obj.slm = self._obj.slm[:, :, to_keep]

        return self

    def copy(self):
        """
        Create an independent copy of HarmoSet

        Returns
        -------
        HarmoSet
            Copy of the object using a new object (deep=True in xarray)

        """
        return HarmoSet(self._obj.copy(deep=True))

    def mean(self):
        # TODO test if ds.xharmo.mean() exist, else write it
        # possibly mean with weight
        pass

    def convolve(self, var):
        """ Convolve over degree
        """
        # TODO test with self._obj.weighted
        # might serve for sh_to_grid()
        pass
