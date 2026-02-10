

from gpu_bSplines import velocity_blueprint
from utils import velocidad


class velocidad_observaciones(velocity_blueprint):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def value(self, x):

        return velocidad(x, self.param)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    x_domain=np.linspace(0,100,100)
    rho_speed=(2650, 1250, 1.5)

    velocidad_prueba=velocidad_observaciones()
    velocidad_prueba.param=rho_speed
    
    plt.figure()
    plt.plot(x_domain,velocidad_prueba.value(x_domain))
    plt.show()
    plt.close()
    pass
