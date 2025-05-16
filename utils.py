
import os
from datetime import datetime
from pickle import dump, load


import math
import numpy as np
from scipy.optimize import Bounds, minimize_scalar
from scipy.special import gamma, kv
from scipy.interpolate import RegularGridInterpolator

# from wave_1 import wave_solver
"""
Este Script va a contener todas las clases y metodos necesarios para el proyecto
"""


class Alpha_min():

    # resticciones para alpha
    # ALPHA_BOUNDS = (Bounds(lb=1e-3, ub=1, keep_feasible=True))
    ALPHA_BOUNDS=(0.,1.)

    def __init__(self, func) -> None:

        self._func = func
        pass

    @property
    def func(self):
        return self._func
    
    # def min_alpha(self, muestra, gradiente_stein, soporte, pool,mg,w,xatol):
    def min_alpha(self,xatol,upper_bound, *args):        
        'esta es la funcion que hay que llamar para optimizar alpha'

        alpha_min = minimize_scalar(fun=self._func,
                            #  x0=x0,
                             args=args,
                             method="Bounded",
                             bounds=(0.,upper_bound),
                            #  bracket=self.ALPHA_BOUNDS,
                            #  xatol=xatol,
                            options={'maxiter': 10, 'disp': 2,'xatol':xatol}
                             )

        return alpha_min


def matern_kernel(r, l, v):
    # r = np.abs(r)
    # r[r == 0] = 1e-8
    # part1 = 2 ** (1 - v) / gamma(v)
    # part2 = (np.sqrt(2 * v) * r / l) ** v
    # part3 = kv(v, np.sqrt(2 * v) * r / l)

    r = np.absolute(r)
    r[r == 0] = 1e-16
    r = np.divide(r, l)
    part1 = 2 ** (1 - v) / math.gamma(v)
    part2 = (math.sqrt(2 * v) * r) ** v
    part3 = kv(v, math.sqrt(2 * v) * r)
    return part1 * part2 * part3


def marginal_variance(l, v, d):
    partn = gamma(v)
    part1 = gamma(v+d/2)
    part2 = (np.pi*4)**(d/2)
    part3 = l**(2*v)
    partd = part1*part2/part3
    return partn/partd


def matern(r, l, v):

    matern_cov = matern_kernel(r, l, v)
    alpha = np.array([marginal_variance(l, v, 1)])

    __matern_cov_prior = np.multiply(matern_cov, alpha)
    __matern_pre_prior = np.linalg.inv(__matern_cov_prior)
    __R = np.linalg.cholesky(__matern_cov_prior)
    __dim = r.shape[0]
    return __matern_cov_prior, __matern_pre_prior, __R, __dim


""" Make sure output directory exists """


def Adam(m0, v0, t, grad_theta0_f, alpha=0.001, b1=0.9, b2=0.999, eps=1e-8):
    t += 1
    mt = b1*m0+(1-b1)*grad_theta0_f
    vt = b2*v0+(1-b2)*(grad_theta0_f**2)
    mhat = mt/(1-b1**t)
    vhat = vt/(1-b2**t)
    step = alpha*mhat/(np.sqrt(vhat)+eps)

    return mt, vt, step


def extraccion_m0(mva: tuple):
    # m,_,_=mva
    return mva[0]


def extraccion_v0(mva: tuple):
    # _,v,_=mva
    return mva[1]


def extraccion_alpha(mva: tuple):
    # _,_,alpha=mva
    return mva[2]


def GradienteProjection_BoxConstrains(x, soporte):

    x_projection = np.copy(x)
    x_projection[x <= soporte[0]] = soporte[0]
    x_projection[x >= soporte[1]] = soporte[1]

    return x_projection


def mkdir(label):
    directory = "resultados/"+label
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory+"/"


def err(x_sample, x_original):
    '''Esta funcion calcula el error relativo'''
    er = np.linalg.norm(x_sample-x_original)/np.linalg.norm(x_original)

    return er


def saveDatosWave(malladoEspacial, malladoTemporal, datos, pathPlusLabel):

    if not (malladoEspacial.shape[0] == datos.shape[0] and malladoTemporal.shape[0] == datos.shape[1]):
        raise ValueError("Las dimensiones no coinciden")
    else:

        malladoEspacialAmpliado = np.hstack((0., malladoEspacial))

        datosAmpliados = np.vstack((malladoTemporal, datos))
        datosAmpliados = np.hstack(
            (malladoEspacialAmpliado[:, np.newaxis], datosAmpliados))

        np.savetxt(pathPlusLabel+".txt", datosAmpliados)


def loadDatosWave(datos):
    malladoEspacial = datos[1:, 0]
    malladoTemporal = datos[0, 1:]
    datos = datos[1:, 1:]

    numero_de_nodos_espaciales = malladoEspacial.size
    numero_de_nodos_temporales = malladoTemporal.size
    dominioEspacial = [malladoEspacial[0], malladoEspacial[-1]]
    dominioTemporal = [malladoTemporal[0], malladoTemporal[-1]]
    dx = malladoEspacial[1]-malladoEspacial[0]
    dt = malladoTemporal[1]-malladoTemporal[0]

    rt = {
        "numero_de_nodos_espaciales": numero_de_nodos_espaciales,
        "numero_de_nodos_temporales": numero_de_nodos_temporales,
        "dx": dx,
        "dt": dt,
        "malladoEspacial": malladoEspacial,
        "malladoTemporal": malladoTemporal,
        "datos": datos,
        "dominioEspacial": dominioEspacial,
        "dominioTemporal": dominioTemporal
    }
    # return malladoEspacial, malladoTemporal,datos
    return rt


def FGP_FD(dx: np.float64, order: int) -> tuple:
    """
    dx: float: es el valor del incremento o resolucion del mallado
    order: int: es el valor de la derivada 1 o 2

    return:
    tupla: devuelve los operadores de derivacion del orden correspondiente asociados a las fronteras
    x0,x1,xi,xn-1,xn
    """

    dx0 = np.zeros(5)
    dx1 = np.zeros(5)
    dxi = np.zeros(5)
    dxn1 = np.zeros(5)
    dxn = np.zeros(5)
    if order == 1:
        # frontera x0
        dx0[0] = -25/(12*dx)
        dx0[1] = 4/dx
        dx0[2] = -3/dx
        dx0[3] = 4/(3*dx)
        dx0[4] = -1/(4*dx)

        # frontera x1
        dx1[0] = -1/(4*dx)
        dx1[1] = -5/(6*dx)
        dx1[2] = 3/(2*dx)
        dx1[3] = -1/(2*dx)
        dx1[4] = 1/(12*dx)

        # xi
        dxi[0] = 1/(12*dx)
        dxi[1] = -2/(3*dx)
        dxi[2] = 0
        dxi[3] = 2/(3*dx)
        dxi[4] = -1/(12*dx)

        # frontera xn-1
        dxn1[0] = -1/(12*dx)
        dxn1[1] = 1/(2*dx)
        dxn1[2] = -3/(2*dx)
        dxn1[3] = 5/(6*dx)
        dxn1[4] = 1/(4*dx)

        # frontera xn
        dxn[0] = 1/(4*dx)
        dxn[1] = -4/(3*dx)
        dxn[2] = 3/dx
        dxn[3] = -4/dx
        dxn[4] = 25/(12*dx)

    elif order == 2:
        # frontera x0
        dx0[0] = 35/(12*dx**2)
        dx0[1] = -26/(3*dx**2)
        dx0[2] = 19/(2*dx**2)
        dx0[3] = -14/(3*dx**2)
        dx0[4] = 11/(12*dx**2)

        # frontera x1
        dx1[0] = 11/(12*dx**2)
        dx1[1] = -5/(3*dx**2)
        dx1[2] = 1/(2*dx**2)
        dx1[3] = 1/(3*dx**2)
        dx1[4] = -1/(12*dx**2)

        # xi
        dxi[0] = -1/(12*dx**2)
        dxi[1] = 4/(3*dx**2)
        dxi[2] = -5/(2*dx**2)
        dxi[3] = 4/(3*dx**2)
        dxi[4] = -1/(12*dx**2)

        # frontera xn-1
        dxn1[0] = -1/(12*dx**2)
        dxn1[1] = 1/(3*dx**2)
        dxn1[2] = 1/(2*dx**2)
        dxn1[3] = -5/(3*dx**2)
        dxn1[4] = 11/(12*dx**2)

        # frontera xn
        dxn[0] = 11/(12*dx**2)
        dxn[1] = -14/(3*dx**2)
        dxn[2] = 19/(2*dx**2)
        dxn[3] = -26/(3*dx**2)
        dxn[4] = 35/(12*dx**2)

    else:
        raise ValueError("No se ha implementado un orden mayor a 2.")

    return dx0, dx1, dxi, dxn1, dxn
# @nb.jit()


def derivative(fx: np.ndarray, dx: np.float64, order: int) -> np.ndarray:
    sx0, sx1, sxi, sxn1, sxn = FGP_FD(dx, order)
    # dxf
    dsfx0 = sx0@fx[:5]
    dsfx1 = sx1@fx[:5]
    dsfxi = np.asarray([sxi@fx[i-2:i+3] for i in range(2, fx.size-2)])
    dsfxn1 = sxn1@fx[-5:]
    dsfxn = sxn@fx[-5:]
    dsfx = np.hstack((dsfx0, dsfx1, dsfxi, dsfxn1, dsfxn))
    return dsfx


def operadorGradiente5P(N, dx):
    dx0, dx1, dxi, dxn1, dxn = FGP_FD(dx, 1)

    gradiente = \
        dxi[0]*np.eye(N, k=-2, dtype=np.float64) +\
        dxi[1]*np.eye(N, k=-1, dtype=np.float64) +\
        dxi[2]*np.eye(N, k=0, dtype=np.float64) +\
        dxi[3]*np.eye(N, k=1, dtype=np.float64) +\
        dxi[4]*np.eye(N, k=2, dtype=np.float64)
    gradiente[:2] = 0
    gradiente[-2:] = 0
    gradiente[0, :5] = dx0
    gradiente[1, :5] = dx1
    gradiente[-2, -5:] = dxn1
    gradiente[-1, -5:] = dxn

    # d1x0, d1x1, d1xi, d1xn1, d1xn = FGP_FD(dx, 1)
    # # dxf
    # d1fx0 = d1x0@fx[:5]
    # d1fx1 = d1x1@fx[:5]
    # d1fx = np.asarray([d1xi@fx[i-2:i+3] for i in range(2, x.size-2)])
    # d1fxn1 = d1xn1@fx[-5:]
    # d1fxn = d1xn@fx[-5:]

    return gradiente


def operadorLaplaciano5P(N, dx):

    dx0, dx1, dxi, dxn1, dxn = FGP_FD(dx, 2)

    Laplaciano = \
        dxi[0]*np.eye(N, k=-2, dtype=np.float64) +\
        dxi[1]*np.eye(N, k=-1, dtype=np.float64) +\
        dxi[2]*np.eye(N, k=0, dtype=np.float64) +\
        dxi[3]*np.eye(N, k=1, dtype=np.float64) +\
        dxi[4]*np.eye(N, k=2, dtype=np.float64)
    Laplaciano[:2] = 0.
    Laplaciano[-2:] = 0.
    Laplaciano[0, :5] = dx0
    Laplaciano[1, :5] = dx1
    Laplaciano[-2, -5:] = dxn1
    Laplaciano[-1, -5:] = dxn

    return Laplaciano


def weights(n, alpha, kappa, beta):
    """
    n es la dimension del espacio de parametros
    alpha hyperparametro
    kappa hyperparametro
    beta hyperparametro

    Wm es el peso para la media
    Wc es el peso para la covarianza
    """

    lambda_ = alpha ** 2 * (n + kappa) - n

    Wc = np.full(2 * n + 1, 1. / (2 * (n + lambda_)))
    Wm = np.full(2 * n + 1, 1. / (2 * (n + lambda_)))
    # voy a modificar esto para que los pesos sumen 1
    Wc[0] = lambda_ / (n + lambda_) + (1. - alpha ** 2 + beta)
    Wm[0] = lambda_ / (n + lambda_)
    return Wm, Wc


def L2_norm(fx, x, axis):
    # esta funcion calcula la norma en L2(0,T)
    return np.trapz(fx**2, x=x, axis=axis)


def sigma_points(X, P, alpha, kappa):
    """
    X es el estado actual
    P es la matriz de covarianza para el estado
    alpha hyperparametro
    kappa hyperparametro
    """
    n = P.shape[0]

    lambda_ = alpha ** 2 * (n + kappa) - n

    sigmas = np.zeros((2 * n + 1, n))
    sigmas[0] = X
    if n == 1:
        U = np.sqrt((n + lambda_) * P)
        for k in range(n):
            sigmas[k + 1] = X + U
            sigmas[n + k + 1] = X - U
    else:
        try:
            U = np.linalg.cholesky((n + lambda_) * P)  # sqrt
        except np.linalg.LinAlgError as err:
            if "Matrix is not positive definite" in str(err):
                return None
        for k in range(n):
            sigmas[k + 1] = X + U[k]
            sigmas[n + k + 1] = X - U[k]

    return sigmas


def wave_solver(wave_speed, nx, nt, wave, fuente):

    w = np.zeros((2, nt, nx))
    for tk in range(1, nt):

        w[0, tk], w[1, tk] = wave.step(ur=w[0, tk-1],
                                       ul=w[1, tk-1],
                                       sur=fuente[tk-1],
                                       sul=0.,
                                       wave_speed=wave_speed)
    return w


def xsmap(x, X, S):

    m = np.diff(S)/np.diff(X)
    n = S[0]-m*X[0]
    return m*x+n


# def soporte_temporal(t0, tf, vel: list or np.ndarray, dx: list or np.ndarray, CFL=0.5):

#     dt = CFL*min(dx)/max(vel)
#     T = (t0, tf)
#     nt = int((T[1]-T[0])/dt)
#     tval, dt = np.linspace(T[0], T[1], nt, retstep=True, endpoint=True)
#     return tval, dt, nt


def soporte_temporal(t0, tf, vel, dx, CFL=0.5):
    '''
    Aqui se calculan
        tval el mallado temporal
        dt la resolucion temporal
        nt numero de nodos del mallado,

    dado un intervalo de tiempo (t0,tf), la resolucion espacial dx y el coeficiente CFL
    '''
    dt = CFL*dx/vel
    T = (t0, tf)
    nt = int((T[1]-T[0])/dt)
    tval, dt = np.linspace(T[0], T[1], nt, retstep=True, endpoint=True)
    return tval, dt


def relacion_senal_ruido(u: list or np.ndarray, percent: float):
    '''
    Aqui se establece una desviacion estandar que cumple con la relacion senal ruido para un tamano de error deseado
    '''
    std = percent*np.abs(u).max()

    return std


def generador_observacion(u: list or np.ndarray, std: float):
    '''
    para generar las observaciones gaussianas se utiliza la relacion
    z=(x-mu)/std
    de donde se obtiene que 
    x=z*std+mu

    siendo:
        z una muestra de una distribucion normal centrada,
        mu media de las observaciones
        std la desviacion estandar deseada
    '''

    ruido = np.random.randn(u.size).reshape(u.shape)

    obs = std*ruido+u
    return obs


def extraer_observaciones_de_xindex(u: list or np.ndarray, xindex: list or np.ndarray):
    '''
    El formato de u es u[tk,x] por tal motivo xindex se aplica sobre el axis=1
    '''
    return u[:, xindex]


def limpiar_directorio(path):
    listdir = os.listdir(path)

    if len(listdir) != 0:
        for file in listdir:
            os.remove(path+'/'+file)

    pass


def list_to_txt(value, path_with_file_name_no_extension):
    """
    Esta funcion es para guardar las observaciones y/o soluciones de la lista como un fichero txt que pueda ser exportador posteriormente a otra aplicacion

    value debe ser una lista en lue todos sus elementos tiene las mismas dimensiones"""

    # out=np.empty((value[0].shape[0],1))
    header = ''
    j_formatter_size = len(str(len(value)))
    for j in range(len(value)):
        xj_formatter_size = len(str(value[j].shape[1]))

        if j == 0:
            out = value[j]
        else:
            out = np.append(out, value[j], axis=1)

        for xj in range(value[j].shape[1]):

            temp = 'R'+str(j).zfill(j_formatter_size)+'X' + \
                str(xj).zfill(xj_formatter_size)
            header += ','+temp

    # guardando el fichero
    np.savetxt(path_with_file_name_no_extension+'.txt',
               out, delimiter=',', header=header[1:])
    pass


def extraer_url_xindex_de_us(valueS, xindex, right_or_left):
    'esta funcion extrae las observaciones en xindex de la solucion en us'

    if right_or_left == 'right' or right_or_left == 0:
        return valueS[0][0, :, xindex]
    elif right_or_left == 'left' or right_or_left == 1:
        return valueS[1][0, :, xindex]
    else:
        raise ValueError


def extraer_ur_xindex_de_solucion(value, xindex):
    '''esta funcion extrae los nodos de observacion de las ondas que viajan hacia la derecha en
    la solucion'''

    ur_xindex = [extraer_url_xindex_de_us(
        value[j], xindex[j], 'right').T for j in range(len(value))]
    return ur_xindex


def extraer_ul_xindex_de_solucion(value, xindex):
    '''esta funcion extrae los nodos de observacion de las ondas que viajan hacia la izquierda en
    la solucion'''
    ul_xindex = [extraer_url_xindex_de_us(
        value[j], xindex[j], 'left').T for j in range(len(value))]
    return ul_xindex


def extraer_xindex_de_solucion(value, xindex):
    '''Esta funcion extrae las observacions de la solucion del modelo directo.

    Value es una lista donde cada una de sus componentes es una 2D-array
    '''
    ur = extraer_ur_xindex_de_solucion(value, xindex)
    ul = extraer_ul_xindex_de_solucion(value, xindex)

    out = [[ur[j], ul[j]] for j in range(len(value))]
    return out


def longitud_onda(vel, freq):
    return vel/freq


def nx_por_longitud_de_onda(longitud_intervalo, velocidad, frecuencia, nodos_por_longitud_de_onda):
    '''esta funcion devuelve el numero de nodos nx por longitud de onda para un intervalo'''

    longitud_onda_value = longitud_onda(velocidad, frecuencia)

    s = longitud_intervalo/longitud_onda_value

    nx = nodos_por_longitud_de_onda*s
    return int(nx)+1


def onda_analitica(c, x, t, std):
    return -(8*std-2*c*t+2*x)*np.exp(-((4*std-c*t+x)/std)**2)/std**2


def gradc(c, x, t, std):
    a = (4*std-c*t+x)
    b = 2*a
    exp_value = np.exp(-((4*std-c*t+x)/std)**2)
    return 2*t*exp_value/std**2-2*t*a*b*exp_value/std**4


def gradx(c, x, t, std):
    a = (4*std-c*t+x)
    b = 2*a
    exp_value = np.exp(-((4*std-c*t+x)/std)**2)
    return -2*exp_value/std**2+b**2*exp_value/std**4


def gradt(c, x, t, std):
    a = (4*std-c*t+x)
    b = 2*a
    exp_value = np.exp(-((4*std-c*t+x)/std)**2)
    return 2*c*exp_value/std**2-2*c*a*2*a*exp_value/std**2


def soporte_to_onda(soporte: dict, objeto,velocidad):
# def soporte_to_onda(soporte: dict, objeto):     #VERSION ORIGINAL
    '''
    Esta funcion recibe un archivo de soporte y devuelve el objeto onda.
    '''
    return objeto(
        **soporte[0], velocidad=velocidad)


def save_to_dat(path_con_nombre_sin_extension, param):
    file = open(path_con_nombre_sin_extension+'.dat', 'wb')
    dump(param, file)
    file.close()
    pass


def load_from_dat(path_con_nombre_sin_extension):
    file = open(path_con_nombre_sin_extension+'.dat', 'rb')
    out = load(file)
    file.close()
    return out


def ab_para_normal_truncada(loc, scale, x0, x1):
    '''esta funcion devuelve los valores de a y b para la distribucion normal truncada centrada en loc y desviacion estandar scale. 
    Aqui los valores de x0 y x1 representan cuales son los limites de los valores deseados'''
    a = (x0-loc)/scale
    b = (x1-loc)/scale
    return a, b


def status(j, n, loss_value):
    # strout=''

    # divido a n en 100 porcientos
    pp = np.quantile(np.arange(n), q=np.linspace(0, 1, 10)).astype(int)
    if j in pp:
        strout = str(round(j/n*100, 2))+'% so far, '+str(n-j) + ' iterations left ' + \
            str(datetime.now())+' loss='+str(round(loss_value, 2))

        print(strout)

def dat_to_json(file,path):
    pass



def parametros_funcion_velocidad(x_domain, p):
    'esta funcion solo desempaquete los parametros necesarios para la funcion de velocidad'
    p0, p1, p2 = p  # desempaquetando los parametros
    domain_length = x_domain[-1]-x_domain[0]  # longitud del intervalo
    return p0, p1, p2, domain_length


def velocidad(x_domain, p):
    '''
    Esta funcion calcula el vector de velocidades al cuadrado asosciado a un dominio x_domain. La velocidad ha de interpretarse como velocidad al cuadrado
    '''
    p0, p1, p2, domain_length = parametros_funcion_velocidad(x_domain, p)

    vel2 = p0+p1*np.sin(2.0*np.pi*x_domain*p2/(domain_length))
    return vel2


def dominio_temporal(dominios_espaciales: list, t_range: list, CFL: float):
    '''esta funcion construye el soporte temporal para todos los subintervalos'''

    t_solver_domain, t_dt = soporte_temporal(
        t0=t_range[0], tf=t_range[1], vel=dominios_espaciales.speed_max, dx=dominios_espaciales.x_dx, CFL=CFL)

    return t_solver_domain, t_dt


def coeficientes_reflexion_transmision(speed_left, speed_right, ns):
    '''
    speed_left es la velocidad en el extremo izquierdo del subintervalo
    speed_right es la velocidad en el extremo derecho del subintervalo
    '''
    R12 = [np.float64(0) for i in range(ns+1)]
    T12 = [np.float64(1) for i in range(ns+1)]
    for i in range(ns-1):
        # coeficientes de reflexion
        R12[i+1] = (speed_left[i+1]-speed_right[i]) / \
            (speed_right[i]+speed_left[i+1])

        # coeficiente de transmision
        T12[i+1] = 2.*speed_left[i+1] / \
            (speed_right[i]+speed_left[i+1])

    return R12, T12


def coeficientes_reflexion_transmision_v02(wave_speed, ns):
    R12 = [np.float64(0) for i in range(ns+1)]
    T12 = [np.float64(1) for i in range(ns+1)]
    for i in range(ns-1):
        # coeficientes de reflexion
        R12[i+1] = (wave_speed[i+1]-wave_speed[i]) / \
            (wave_speed[i] + wave_speed[i+1])

        # coeficiente de transmision
        T12[i+1] = 2.*wave_speed[i+1] / \
            (wave_speed[i] + wave_speed[i+1])

    return R12, T12

def rho_params_generator(lower_speed, upper_speed):

    p0 = np.random.randint(low=lower_speed, high=upper_speed)

    p1 = np.min((upper_speed-p0, p0-lower_speed))

    p2 = np.random.randint(low=1, high=10)

    return [p0, p1, p2]

def interpolator(xp_values: list or np.ndarray, tp_values: list or np.ndarray, x_mesh_values: list or np.ndarray, t_mesh_values: list or np.ndarray, data_mesh_values: np.ndarray, method: str = 'linear'):
    '''
    xp_values: son los valores espaciales sobre los cuales se desea interpolar
    tp_values: son los valores temporales sobre los cuales se desea interpolar
    x_mesh_values: son los valores espaciales en los cuales estan contenidas las solucions de data_mesh_values
    t_mesh_values: son los valores temporales en los cuales estan contenidas las soluciones de data_mesh_values
    data_mesh_values: son los valores de la solucion para las coordenadas x_mesh y t_mesh


    RETURN:
    Devuelve el valor de la interpolacion en con forma (xp_values.shape,tp_values.shape)
    '''

    interp = RegularGridInterpolator(points=(t_mesh_values, x_mesh_values),
                                     values=data_mesh_values,
                                     bounds_error=True,
                                     method=method)

    XP, TP = np.meshgrid(xp_values, tp_values, indexing='xy')

    interpolate_values = interp((TP, XP))
    return interpolate_values.T

def wave_obs_extract(layers, x_obs_list, t_obs):
    wave_obs = [interpolator(xp_values=layers[j].x_domain[x_obs_list[j]],
                             tp_values=t_obs,
                             x_mesh_values=layers[j].x_domain,
                             t_mesh_values=layers[j].t_domain,
                             data_mesh_values=layers[j].wave_amplitude.T)
                for j in range(len(layers))]
    
    return wave_obs

if __name__ == "__main__":

    dx=0.000001
    order=1
    n=7
    
    operator=operadorGradiente5P(N=n,dx=dx)
    print(operator.round())
    pass
