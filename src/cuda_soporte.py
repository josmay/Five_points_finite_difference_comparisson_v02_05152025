import cupy as cp
import numpy as np
# from src.on_hold.cuda_soporte import Soporte
from cupyx.scipy.interpolate import interpn
from scipy import signal, stats

from gpu_bSplines import velocity_field
from finite_difference_operators import fp_operator
from utils import soporte_temporal


class Soporte():

    # variable para almacenar las observaciones
    __data = None
    __std2 = None
    __std_percent = 0.01
    __dervidada_params=None
    # nParams = 5  # numero de parametos del modelo de betasplines
    # kParams = 1  # numero de grados de libertad para le modelo de betasplines
    # nTobs = 100  # numero de nodos temporales de observacion

    def __init__(
        self, X_range, lambda_nodes, source_frequency, speed_range,rho_param, T_range, CFL, nParams=5, kParams=1, nTobs=100
    ) -> None:

        self.nParams, self._kParams, self.nTobs = nParams, kParams, nTobs
        self.rho_param=rho_param
        # Aqui configuramos el dominio con todos los elementos necesarios para los nodos temporales y espaciales. Estos nodos y valores satisfacen la condicion CFL
        self.__x_range = X_range  # limites del intervalo espacial

        # numero de nodos espaciales por cada longitud de onda minima
        self.__lambda_nodes = lambda_nodes
        self.__source_frequency = source_frequency

        self.__speed_range = speed_range  # min y max del campo de velocidades
        # determinando el numero de nodos espaciales

        # para calcular x_nodes
        self.__nodos()
        # soporte numerico en X, la resolucion dx
        self.__x_domain, self.__x_dx = cp.linspace(
            *self.__x_range, num=self.x_nodes, retstep=True
        )

        # construccion de las matrices para las derivadas
        self._Gx5p = cp.array(fp_operator(self.__x_dx, 1, self.x_nodes))
        self._Lx5p = cp.array(fp_operator(self.__x_dx, 2, self.x_nodes))
        # self._Gx5p = cp.array(operadorGradiente5P(
        #     N=self.x_nodes, dx=self.__x_dx))
        # self._Lx5p = cp.array(operadorLaplaciano5P(
        #     N=self.x_nodes, dx=self.__x_dx))

        # dominio temporal de todo el soporte
        self.__t_domain, self.__t_dt = soporte_temporal(
            t0=T_range[0], tf=T_range[1], vel=self.speed_max, dx=self.__x_dx, CFL=CFL
        )

        self.__t_domain = cp.array(self.__t_domain)
        self.__t_dt = cp.array(self.__t_dt)

        # aqui se construyen los atributos para las observaciones
        self.__nodos_de_observacion()

        # fuente del modelo directo en la frontera izquierda
        self.__source_left = signal.gausspulse(
            self.t_domain - self.t_domain[-1] * 0.25, fc=self.source_frequency
        )

        # velocidad asociada al modelo
        self.__velocidad = velocity_field(
            k=self._kParams, n=self.nParams, x_domain=self.x_domain)

    @property
    def kParams(self):
        return self._kParams
    
    @kParams.setter
    def kParams(self,new):
        self._kParams=new
        
        # velocidad asociada al modelo
        self.__velocidad = velocity_field(
            k=self._kParams, n=self.nParams, x_domain=self.x_domain)
    
    
    def __nodos_de_observacion(self):
        # nodos espaciales de observacion
        self.__x_obs_list = np.quantile(
            np.arange(self.x_nodes), q=(np.linspace(0.015, 0.995, self.nParams))
        ).astype(int)

        self.__xp_values = cp.array(self.x_domain[self.__x_obs_list])

        # observaciones del dominio temporal
        self.__tp_values = cp.linspace(
            self.t_domain[0], self.t_domain[-1], self.nTobs)

        self.__meshgrid_XP_TP()

    def __nodos(self):
        # numero de nodos espaciales en funcion de la frecuencia, la velocidad y la longitud de onda
        # numero de nodos para el dominio espacial
        x_nodes = np.diff(self.__x_range) / \
            self.wavelength_min * self.__lambda_nodes
        self.__x_nodes = int(np.ceil(x_nodes)[0])
        # return x_nodes

    def __meshgrid_XP_TP(self):
        # Create the meshgrid for xp_values and tp_values
        TP, XP = cp.meshgrid(self.tp_values, self.xp_values, indexing='ij')

        # Prepare the points for interpn by stacking the XP and TP arrays
        self.__points = cp.stack((XP.ravel(), TP.ravel()), axis=-1)

    @property
    def data(self):
        return self.__data

    # @data.setter
    def data_method(self, solucion_onda, std_percent=0.01,seed=None):

        if self.__std_percent != std_percent or self.__data is None:
            wave_obs = self.wave_obs_extract(solucion_onda[:, :self.x_nodes,:])
            # relacion señal ruido de los datos
            # porcentaje en la relacion senal a ruido
            std = std_percent * cp.abs(wave_obs).max()
            self.__data = cp.array(stats.norm.rvs(loc=wave_obs.get(), scale=std.get(),random_state=seed))
            self.__std2 = cp.square(std)

    @property
    def std2(self):
        return self.__std2

    def wave_obs_extract(self, data_mesh_values):
        """
        Extract wave observations over a batch of data using vectorized operations, handling different sizes of tp_values and xp_values.

        Parameters:
        xp_values: array-like
            Spatial observation points.
        tp_values: array-like
            Temporal observation points.
        x_mesh_values: array-like
            The spatial domain mesh grid values.
        t_mesh_values: array-like
            The temporal domain mesh grid values.
        data_mesh_values: cupy.ndarray
            The data to interpolate, with shape (batch_size, nx, nt).

        Returns:
        cupy.ndarray
            Interpolated values with shape (batch_size, len(tp_values), len(xp_values)).
        """
        if data_mesh_values.ndim <= 2:
            data_mesh_values = data_mesh_values[cp.newaxis, :, :]
        batch_size = data_mesh_values.shape[0]

        # Perform the vectorized interpolation for the entire batch
        wave_obs = cp.empty(
            (batch_size, len(self.tp_values), len(self.xp_values)))

        for i in range(batch_size):
            # Interpolate for the current batch
            wave_obs[i] = interpn(
                (self.x_domain, self.t_domain),
                data_mesh_values[i],
                self.points,
                method='linear',
                bounds_error=False,
                fill_value=None
            ).reshape(len(self.tp_values), len(self.xp_values))

        return wave_obs

    # cp.fuse(kernel_name='L2norm')
    def L2norm(self, y):
        "Norma L2 en el espacio de observaciones"

        if y.ndim <= 2:
            y = y[cp.newaxis, :]

        "output variable"
        # out=cp.empty_like(y)

        cp.square(y, out=y)
        y = cp.trapz(y, x=self.tp_values, axis=1)
        cp.sqrt(y, out=y)
        # return y

    def speed_square(self, x):
        return self.velocidad.value(x)
    
    @property
    def speed_square_grad_prams(self):
        
        #este valor es constante debido al diseño del modelo. Por eso si incluye aqui
        if self.__dervidada_params is None:
            self.__dervidada_params=self.velocidad.basis_spline_vector(self.xp_values)
        return self.__dervidada_params

    @property
    def points(self):
        return self.__points

    @property
    def velocidad(self):
        return self.__velocidad

    @property
    def source_left(self):
        return self.__source_left

    @property
    def x_obs_list(self):
        return self.__x_obs_list

    @property
    def xp_values(self):
        return self.__xp_values

    @property
    def tp_values(self):
        return self.__tp_values

    @property
    def x_nodes(self):
        return self.__x_nodes

    @property
    def wavelength_min(self):
        return self.speed_min / self.__source_frequency

    @property
    def wavelength_max(self):
        return self.speed_max / self.__source_frequency

    @property
    def speed_max(self):
        return self.__speed_range[1]

    @property
    def speed_min(self):
        return self.__speed_range[0]

    @property
    def Lx5p(self):
        """Devuelve el operador Laplaciano de derivada de 5 puntos"""
        return self._Lx5p

    @property
    def Gx5p(self):
        """Devuelve el operador Gradiente de derivada de 5 puntos"""
        return self._Gx5p

    @property
    def X_range(self):
        return self.__x_range

    @property
    def lambda_nodes(self):
        return self.__lambda_nodes

    @property
    def source_frequency(self):
        return self.__source_frequency

    @property
    def t_domain(self):
        return self.__t_domain

    @property
    def t_dt(self):
        return self.__t_dt

    @property
    def x_domain(self):
        return self.__x_domain

    @property
    def x_dx(self):
        return self.__x_dx

    @property
    def speed_range(self):
        return self.__speed_range


def muestrasHalton(nSamples: int, nParams: int, soporte: Soporte, seed=None):

    from scipy.stats import qmc

    # % valores iniciales. Valores tipo Halton
    halton_muestras = qmc.Halton(d=nParams, seed=seed)
    muestra = halton_muestras.random(nSamples)
    # variables que van a almacenar todos los resultados
    gpu_muestra_vector_pool = cp.array(
        qmc.scale(muestra, soporte.speed_min**2, soporte.speed_max**2)
    )

    return gpu_muestra_vector_pool


if __name__ == "__main__":
    from gpu_bSplines import velocity_field
    from velocidad_observaciones import velocidad_observaciones

    # velocidad={"objeto":velocidad_observaciones,
    #            "kwargs":{
    #            }}

    __args = {
        "X_range": (0.0, 100.0),
        "lambda_nodes": 10,
        "source_frequency": 10,
        "speed_range": [37.0, 63.0],
        "T_range": np.array([0.0, 4.0], dtype="float32"),
        "CFL": 0.25
    }

    soporte = Soporte(**__args)

    pass
