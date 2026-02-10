
import cupy as cp
import numpy as np

from cuda_soporte import Soporte

class WaveModeloSolucion:
    def __init__(self, soporte: Soporte) -> None:

        self.soporte = soporte
        # Aqui configuramos el dominio con todos los elementos necesarios para los nodos temporales y espaciales. Estos nodos y valores satisfacen la condicion CFL
        pass

    def extraer_u_v(self, w):
        "este metodo extrae de w a u y a v=dtu"
        u = w[:, :self.soporte.x_nodes]
        v = w[:, self.soporte.x_nodes:]
        return u, v

    def frontera_absorvente_derecha(self, u1, dxn, speed_n):
        return -speed_n * cp.dot(u1[:, -5:], dxn)

    def frontera_absorvente_izquierda(self, u1, dx0, speed_0):
        return speed_0 * cp.dot(u1[:, :5], dx0)

    def operador_u(self, vn1, un0):
        return 2 * self.soporte.t_dt * vn1 + un0

    def operador_v(self, u1, v1, v0, speed_square, speed_square_grad):
        
        #Aplicando el teorema de la divergencia
        # div(c^2(x)Grad(u(x)))=grad(c^2(x)).grad(u(x))+c^2(x)*Lap(u(x))

        # derivada
        Gx5pU = cp.squeeze(self.soporte.Gx5p @ u1[:, :, cp.newaxis], axis=-1)

        # laplaciano
        Lx5pU = cp.squeeze(
            self.soporte.Lx5p[cp.newaxis, :, :] @ u1[:, :, cp.newaxis], axis=-1)
        return (2 * self.soporte.t_dt) * (
            speed_square_grad * (Gx5pU) + speed_square * (Lx5pU)
        ) + v0

    def operador_de_onda(
        self, w1, w0, speed_square, speed_square_grad
    ):
        "aqui se soluciona el sistema de ecuaciones asociado al modelo"

        u1, v1 = self.extraer_u_v(w1)
        u0, v0 = self.extraer_u_v(w0)

        u_next = self.operador_u(v1, u0)
        v_next = self.operador_v(
            u1, v1, v0, speed_square, speed_square_grad
        )
        return u_next, v_next

    def step(self, w1, w0, speed_square, speed_square_grad):
        """
        Esta funcion calcula el proximo paso en la solucion de la ecuacion de onda.
        """
        u1, _ = self.extraer_u_v(w1)
        u_next, v_next = self.operador_de_onda(
            w1,
            w0,
            speed_square,
            speed_square_grad
        )

        # condicion de frontera absorvente
        v_next[:, 0] = self.frontera_absorvente_izquierda(
            u1, self.soporte.Gx5p[0, :5], cp.sqrt(speed_square[:, 0])
        )
        v_next[:, -1] = self.frontera_absorvente_derecha(
            u1, self.soporte.Gx5p[-1, -5:], cp.sqrt(speed_square[:, -1])
        )

        return u_next, v_next

    # @time_execution
    def solver(self, speed_square, speed_square_grad):
        "Esta funcion calcula la solucion a la ecuacion de onda utilizando GPU."

        batch_size = speed_square.shape[0]
        n_time_steps = len(self.soporte.t_domain)
        n_points = 2 * self.soporte.x_nodes
        out = cp.zeros((batch_size, n_points, n_time_steps), dtype="float32")

        # Set initial condition
        out[:, 0, 0] = self.soporte.source_left[0]

        for t_index in range(n_time_steps-1 ):
            
            out[:,0, t_index] = self.soporte.source_left[t_index]
            out[:,:self.soporte.x_nodes, t_index+1], out[:,self.soporte.x_nodes:, t_index+1] = self.step(out[:,:, t_index],out[:,:, t_index-1],speed_square,speed_square_grad)

        return out


if __name__ == "__main__":

    import os
    from time import time

    import matplotlib.pyplot as plt

    import plots_forward_v01
    from cuda_soporte import muestrasHalton
    from velocidad_observaciones import velocidad_observaciones

    # test_keys = ('GSVGD_ADAM_w0.5', 'GSVGD_ADAM_w0.0', 'GSVGD_ADAM_w1.0')
    test_keys = ["SYB-CUDA-FWM-20240708"]

    paths = {
        "load_path": os.path.abspath("resultados/fwm"),
        "mcmc_path": os.path.abspath("resultados/mcmc_twalk"),
        test_keys[0]: os.path.abspath("resultados/" + test_keys[0]),
    }  # ,
    #  test_keys[1]: os.path.abspath('resultados/'+test_keys[1]),
    #  test_keys[2]: os.path.abspath('resultados/'+test_keys[2])}

    # paths
    for key, values in paths.items():
        if not os.path.exists(paths[key]):
            os.makedirs(paths[key])

    # lista de los argumentos para construir los modelos continuos en cada una de las capas del modelo surrogate
    __args = {
        "X_range": (0.0, 100.0),
        "lambda_nodes": 10,
        "source_frequency": 10,
        "speed_range": [37.0, 63.0],
        "T_range": np.array([0.0, 4.0], dtype="float32"),
        "CFL": 0.25,
        "rho_param":(2000, 500, 1.)}

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
    speed_square_00=cp.ones_like(speed_square_00)
    speed_square_grad_00 = soporte.Gx5p @ speed_square_00
    speed_square_grad_00=cp.zeros_like(speed_square_grad_00)

    wave_sintetico = solver_modelo_directo.solver(
        speed_square=speed_square_00[cp.newaxis, :],
        speed_square_grad=speed_square_grad_00[cp.newaxis, :]
    )

    soporte.data_method(wave_sintetico, 0.05)

    plots_forward_v01.app(soporte.data.get(), soporte.xp_values.get(
    ), soporte.tp_values.get(), paths['load_path'])

    ##############################################################
    ##############################################################
    ##############################################################

    # % valores iniciales. Valores tipo Halton
    gpu_muestra_vector_pool = muestrasHalton(
        nSamples, nParams, soporte, seed=1992)

    gpu_speed_square = cp.zeros((nSamples, soporte.x_nodes), dtype='float')
    gpu_speed_square_grad = cp.zeros_like(gpu_speed_square)

    for xi in range(nSamples):
        soporte.velocidad.param = gpu_muestra_vector_pool[xi]

        gpu_speed_square[xi] = soporte.velocidad.value(soporte.x_domain)
        gpu_speed_square_grad[xi] = soporte.Gx5p @ gpu_speed_square[xi]

    # tiempo cpu
    test = solver_modelo_directo.solver(
        speed_square=gpu_speed_square,
        speed_square_grad=gpu_speed_square_grad
    )

    # cuda_solver_modelo_directo(gpu_source_left,gpu_speed_square,gpu_speed_square_grad,gpu_Gx5p,gpu_Lx5p,gpu_t_dt,gpu_t_domain,soporte.x_nodes)

    # wave_solution=cp.asnumpy(gpu_wave_solution)

    plt.figure()
    # for t in range(len(soporte.t_domain)):
    plt.plot(soporte.x_domain.get(),
             test[nSamples//4, :soporte.x_nodes, :].get()[:soporte.x_nodes])
    plt.title('X/T')
    plt.show()
    plt.close()
