
"""
Este script correra la solucion al modelo de ecuacion de ondas utilizando el operador de 5 puntos para la aproximacion de las derivadas en el metodo de diferencias finitas.
"""
import os

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuda_modelo_directo_5P_open_boundary import WaveModeloSolucion
from cuda_soporte import Soporte
# from cuda_soporte import muestrasHalton
from velocidad_observaciones import velocidad_observaciones

paths = {
    "load_path": os.path.abspath("resultados/fwm"),
    "mcmc_path": os.path.abspath("resultados/mcmc_twalk"),
    # test_keys[0]: os.path.abspath("resultados/" + test_keys[0]),
    # test_keys[1]: os.path.abspath("resultados/" + test_keys[1]),
    # test_keys[2]: os.path.abspath("resultados/" + test_keys[2]),
}


def FD_5p(lambda_nodes):
    __args = {
            "X_range": (0.0, 100.0),
            "lambda_nodes": lambda_nodes,
            "source_frequency": 10,
            "speed_range": [37.0, 63.0],
            "T_range": np.array([0.0, 4.0], dtype="float32"),
            "CFL": 0.25,
            "rho_param": (2000, 500, 1.)}

        # PARAMETROS DEL MODELO
    nParams = 5
    kParams = 0
    nTobs = 500
    nSamples = 200

    soporte = Soporte(**__args, nParams=nParams, kParams=kParams, nTobs=nTobs)

    solver_modelo_directo = WaveModeloSolucion(soporte)

        # Datos Sinteticos
        # estos datos corresponden a la velocidad al cuadrado
    rho_speed = (2650, 1250, 1.5)
    velocidad_prueba = velocidad_observaciones()
    velocidad_prueba.param = rho_speed

    speed_square_00 = velocidad_prueba.value(soporte.x_domain)
    speed_square_00 = cp.array(speed_square_00)
    speed_square_grad_00 = soporte.Gx5p @ speed_square_00
    # speed_square_grad_00 = cp.zeros_like(speed_square_grad_00)

    wave_sintetico = solver_modelo_directo.solver(
            speed_square=speed_square_00[cp.newaxis, :],
            speed_square_grad=speed_square_grad_00[cp.newaxis, :]
        )

        # cuda_solver_modelo_directo(gpu_source_left,gpu_speed_square,gpu_speed_square_grad,gpu_Gx5p,gpu_Lx5p,gpu_t_dt,gpu_t_domain,soporte.x_nodes)

        # wave_solution=cp.asnumpy(gpu_wave_solution)

        # plt.figure()
        # # for t in range(len(soporte.t_domain)):
        # plt.plot(soporte.x_domain.get(),
        #          wave_sintetico[0, :soporte.x_nodes,:].get())
        # plt.title('X/T')
        # plt.show()
        # plt.close()
        
        # plt.figure()
        # # for t in range(len(soporte.t_domain)):
        # plt.plot(soporte.t_domain.get(),
        #          wave_sintetico[0, :soporte.x_nodes,:].T.get()[:,[-5]])
        # plt.title('X/T')
        # plt.show()
        # plt.close()
        
        
    z_norm=cp.asnumpy(wave_sintetico[0,:soporte.x_nodes,:][-1]-wave_sintetico[0,:soporte.x_nodes,:][-1].mean())
    return z_norm

if __name__ == "__main__":

    
    norma=[]
    
    
    for lambda_nodes in range(5,50,5):
        # lista de los argumentos para construir los modelos continuos en cada una de las capas del modelo surrogate
        z_norm = FD_5p(lambda_nodes)
        
        norma.append(z_norm)
        

    std=[np.std(i,ddof=-1) for i in norma]

    x_vals = range(5, 50, 5)
    plt.plot(x_vals, std, 'r.', label="Red Markers")  # Red markers only
    plt.plot(x_vals, std, 'b-', label="Blue Line")    # Blue line only

    # Adding title and labels
    plt.title("Plot with Red Markers and Blue Line")
    plt.xlabel("X-axis")
    plt.ylabel("std values")
    plt.legend()
    plt.show()
    pass
