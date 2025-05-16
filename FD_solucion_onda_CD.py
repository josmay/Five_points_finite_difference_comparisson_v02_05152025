
"""
Este script correra la solucion al modelo de ecuacion de ondas utilizando el operador tradicional (2 y 3 puntos) para la aproximacion de las derivadas en el metodo de diferencias finitas.
"""

import os

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from cuda_modelo_directo_CD_open_boundary import WaveModeloSolucion
from cuda_soporte import Soporte
# from cuda_soporte import muestrasHalton
from velocidad_observaciones import velocidad_observaciones

paths = {
    "load_path": os.path.abspath("resultados/fwm"),
    "mcmc_path": os.path.abspath("resultados/mcmc_twalk"),
}


def FD_CD(lambda_nodes):
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
    # nSamples = 200

    soporte = Soporte(**__args, nParams=nParams, kParams=kParams, nTobs=nTobs)
        
        #modificando los operadores de diferencias finitas
        
        #Diferencias hacia adelante 1er orden
        
    nodes=soporte.x_nodes
    # nodes=5    
    
    dx=soporte.x_dx
    
    Gx, Lx = FD_CD_operadores(nodes, dx)
        
        #diferencias finitas en el soporte
    soporte._Gx5p=Gx
    soporte._Lx5p=Lx
        

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

    plt.figure()
    # for t in range(len(soporte.t_domain)):
    plt.plot(soporte.x_domain.get(),
                wave_sintetico[0, :soporte.x_nodes,:].get())
    plt.title('X/T')
    plt.show()
    plt.close()
    
    plt.figure()
    # for t in range(len(soporte.t_domain)):
    plt.plot(soporte.t_domain.get(),
                wave_sintetico[0, :soporte.x_nodes,:].T.get()[:,[-5]])
    plt.title('X/T')
    plt.show()
    plt.close()

    z_norm=cp.asnumpy(wave_sintetico[0,:soporte.x_nodes,:][-1]-wave_sintetico[0,:soporte.x_nodes,:][-1].mean())
    return z_norm

def FD_CD_operadores(nodes, dx):
    Gx= cp.diag(np.tile(-1,nodes-1),k=-1)+cp.diag(np.tile(1,nodes-1),k=1)
    Gx=Gx/(2*dx)
    Gx[0,:2]=cp.array([-1,1])/dx
    Gx[-1,-2:]=cp.array([-1,1])/dx

    #Diferencias hacia adelante 2do orden
    Lx=cp.diag(np.tile(1,nodes-1),k=-1)+cp.diag(np.tile(-2,nodes),k=0)+cp.diag(np.tile(1,nodes-1),k=1)
    Lx[0,:2]=cp.array([-2,2])
    Lx[-1,-2:]=cp.array([2,-2])
    Lx=Lx/dx**2
    return Gx,Lx

if __name__ == "__main__":


    norma=[]
    
    x_vals = range(5, 50, 5) 
    for lambda_nodes in x_vals:
    # lista de los argumentos para construir los modelos continuos en cada una de las capas del modelo surrogate
        z_norm = FD_CD(lambda_nodes)
        
        
        norma.append(z_norm)
        

    std=[np.std(i,ddof=-1) for i in norma]

    
    plt.plot(x_vals, std, 'r.', label="Red Markers")  # Red markers only
    plt.plot(x_vals, std, 'b-', label="Blue Line")    # Blue line only

    # Adding title and labels
    plt.title("Plot with Red Markers and Blue Line")
    plt.xlabel("X-axis")
    plt.ylabel("std values")
    plt.legend()
    plt.show()
