

import abc
from time import sleep

# from scipy.interpolate import BSpline
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
# import scipy.integrate
from cupyx.scipy.interpolate import BSpline

from utils import operadorGradiente5P, operadorLaplaciano5P, velocidad


class velocity_blueprint():

    def __init__(self, *args, **kwargs) -> None:
        pass

    @property
    def param(self):
        return self._param

    @param.setter
    def param(self, new):
        self._param = new

    def value(self, x):
        pass


class velocity_field(velocity_blueprint):

    nu = 1  # orden de la derivada espacial del spline

    def __init__(self, k: int, n: int, x_domain: np.ndarray) -> None:
        """
        k: grado de la base del spline
        n: numero de coeficientes
        x_domain: soporte numerico para establecer los nots
        """

        assert (n >= k+1)

        self.k = k  # grado de la base
        self.n = n  # numero de parametros

        self.n_knots = self.n+self.k+1  # tamaño del vector de knots

        self.gpu_knots = self.knots(x_domain)

        c_auxiliar = cp.eye(self.n)

        self.auxiliar_splines = [
            BSpline(t=self.gpu_knots, c=c_auxiliar[i], k=self.k) for i in range(self.n)]

        del c_auxiliar

    def basis_spline_vector(self, x):
        """
        Esta funcion calcula la derivada del spline respecto a los coeficiente. Devuelve un vector de longitud (n,) con el valor de cada base B_i,k(x)

        el resultado es como sigue, cada columna representa el valor de la derivadad para cada valor de x.
        Asi la columna 0 es la d/dc(x0), etc col i -> d/dc(xi)
        """

        return cp.array([bi(x) for bi in self.auxiliar_splines])

    def knots(self, x_domain):

        gpu_knots = cp.zeros(self.n_knots)
        gpu_knots[self.k:self.n +
                  1] = cp.linspace(*x_domain[[0, -1]], self.n+1-(self.k))
        gpu_knots[self.n+1:] = cp.linspace(gpu_knots[self.n],
                                           gpu_knots[self.n]*1.01, self.n_knots-(self.n+1))

        return gpu_knots

    @property
    def basis_knots(self):
        return self.gpu_knots[self.k:self.n+1]

    @property
    def param(self):
        return self._param

    @property
    def beta_spline(self):
        return self._beta_spline

    @param.setter
    def param(self, new_param):

        self._param = new_param

        self._beta_spline = BSpline(
            t=self.gpu_knots, c=self._param, k=self.k, extrapolate=False)

        pass

    def value(self, x):
        # self.param=param
        return self.beta_spline(x)


if __name__ == '__main__':

    runs_modelo_continuo = {'fwm': {'X_range': (0., 100.),
                                    'T_range': (0., 2.5),
                                    'lambda_nodes': 150,
                                    'CFL': 0.25,
                                    'rho_speed': (2650, 1250, 1),
                                    'source_frequency': 10}}

    x_domain, dx = np.linspace(
        *runs_modelo_continuo['fwm']['X_range'], num=100, retstep=True)

    G5p = operadorGradiente5P(100, dx=dx)
    L5p = operadorLaplaciano5P(100, dx=dx)

    velocidad_cuadrado = velocidad(
        x_domain=x_domain, p=runs_modelo_continuo['fwm']['rho_speed'])

    velocity_object = velocity_field(k=1, n=20, x_domain=x_domain)
    velocity_object.param = cp.array(
        velocidad_cuadrado[::len(velocidad_cuadrado)//(20-1)])

    plt.figure()
    plt.plot(velocity_object.basis_spline_vector(x_domain).get().T)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(x_domain, velocidad_cuadrado, label='velocidad', color='blue')
    plt.plot(x_domain, velocity_object.value(x_domain).get(),
             label='Object', color='red', ls=':')
    plt.legend(loc='upper left')
    plt.show()
    plt.close()

    pass
