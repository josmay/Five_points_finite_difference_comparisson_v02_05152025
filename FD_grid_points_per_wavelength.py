"""
Este script muestra la convergencia de la aproximacion de la derivada utilizando el operador de 5 puntos.
"""


import matplotlib.pyplot as plt
import numpy as np

from five_point_operator import fp_operator
from pictures_setup import dpi, figsize, fontsize
from utils import err


def examples(fun, dxfun, dxxfun, domain, k_values, err_out):
    for k_iter in range(k_values.size):
        x, dx = np.linspace(domain[0][0], domain[0]
                            [1], k_values[k_iter], retstep=True)

        # derivative of the functions using the algebraic expression
        fx = fun(x)
        dxfx = dxfun(x)
        dxxfx = dxxfun(x)

        # derivative of the functions using the five points operator
        # 1st derivative
        d1_new = fp_operator(size=k_values[k_iter], n=1, dx=dx)
        d1fx = d1_new@fx

        # 2nd derivative
        d2_new = fp_operator(size=k_values[k_iter], n=2, dx=dx)
        d2fx = d2_new@fx

        err_out[k_iter] = err(d1fx, dxfx), err(d2fx, dxxfx)
    return err_out


if __name__ == "__main__":
    import os

    savepath = "pictures/finite_differences"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    alpha_error = 0.0005  # alpha_error*100=0.05%

    # ejemplo 1 Funcion Acos(wt-kx+phi)
    A = 1.  # amplitud de la onda sinusoidal
    freq = 0.0001  # frecuencia de la onda sinusoidal
    phi = -0.5*np.pi  # phase de la onda
    w = 2*np.pi*freq  # frecuencia angular
    T = 1/freq  # periodo de la onda
    wavelengh = 1*T  # longitud de la onda
    wavenumber = 2*np.pi/wavelengh
    t = 0

    def fun1(x): return A*np.cos(w*t-wavenumber*x)
    def dxfun1(x): return A*wavenumber*np.sin(w*t-wavenumber*x)
    def dxxfun1(x): return -A*wavenumber**2*np.cos(w*t-wavenumber*x)

    # ejemplo 2 Funcion exp(-x**2/(2*std**2))
    # std = 3/16*wavelengh
    s = 4
    std = wavelengh/s
    def fun2(x): return np.exp(-1/(2*std**2)*(x)**2)/(np.sqrt(2*np.pi)*std)

    def dxfun2(x): return -x/(std**2) * \
        np.exp(-1/(2*std**2)*x**2)/(np.sqrt(2*np.pi)*std)
    def dxxfun2(x): return (-1/std**2+x**2/std**4) * \
        np.exp(-1/(2*std**2)*x**2)/(np.sqrt(2*np.pi)*std)

    # ejemplo 3 Funcion Asin(wx+phi)+exp(-x**2/(2*std**2))/(np.sqrt(2*np.pi)*std)

    def fun3(x): return fun1(x)+fun2(x)
    def dxfun3(x): return dxfun1(x)+dxfun2(x)
    def dxxfun3(x): return dxxfun1(x)+dxxfun2(x)

    domain1 = [-0.5*wavelengh, 0.5*wavelengh]
    domain2 = [-s*std, s*std]
    domain3 = [np.minimum(domain1[0], domain2[0]),
               np.maximum(domain1[1], domain2[1])]
    domain = np.array([domain1, domain2, domain3])

    # these are the functions and they 1st and 2nd order derivatives
    fun = [fun1, fun2, fun3]
    dxfun = [dxfun1, dxfun2, dxfun3]
    dxxfun = [dxxfun1, dxxfun2, dxxfun3]

    k_values = np.arange(5, 100)

    errors = [np.zeros((k_values.size, 2)),  # ejemplo 1 sin(x)
              # ejemplo 2 np.exp(-1/(2*std**2)*x**2)
              np.zeros((k_values.size, 2)),
              # ejemplo 3 Campana de Gauss np.sin(x)+np.exp(-1/(2*std**2)*x**2)
              np.zeros((k_values.size, 2))]

    errors = [examples(fun[i], dxfun[i], dxxfun[i], domain,
                       k_values, errors[i]) for i in range(len(errors))]

    kmin_index = np.zeros(shape=(3, 2), dtype=int)
    kmin_index[0] = np.argmin(np.abs(errors[0]-alpha_error), axis=0)
    kmin_index[1] = np.argmin(np.abs(errors[1]-alpha_error), axis=0)
    kmin_index[2] = np.argmin(np.abs(errors[2]-alpha_error), axis=0)

    # parametros para los graficos
    # xticks
    e1_xticks = [-0.5*wavelengh, 0, 0.5*wavelengh]
    e2_xticks = [-s*std, 0, s*std]

    if 5*std > domain1[1]:
        e3_xticks = [-s*std,
                     domain1[0],
                     0,
                     domain1[1],
                     s*std]
        e3_xticks_label = [r"$-4\sigma$",
                           r"$-\lambda /2$",
                           r"$0$",
                           r"$\lambda /2$",
                           r"$4\sigma$"]
    else:
        e3_xticks = [domain1[0],
                     -s*std,
                     0,
                     s*std,
                     domain1[1]]
        e3_xticks_label = [r"$-\lambda /2$", r"$-4\sigma$",
                           r"$0$", r"$4\sigma$", r"$\lambda /2$"]

    xticks_fig2 = [e1_xticks, e2_xticks, e3_xticks]

    e1_xticks_label = [r"$-\lambda /2$", r"$0$", r"$\lambda /2$"]
    e2_xticks_label = [r"$-4\sigma$", r"$0$", r"$4\sigma$"]
    xticks_label_fig2 = [e1_xticks_label, e2_xticks_label, e3_xticks_label]

    # linestyles and markers
    fxls = ["solid", "dotted", "dashdot"]
    FDmarker = ["x", "."]

    # legend
    labels_fig1 = [r"$FD-\partial_x f$",
                   r"$FD-\partial_x^2 f$", r"$\varepsilon=0.0005$"]

    # titulos
    titles = [r"$f\left(x\right)=A\cos(kx)$", r"$f\left(x\right)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\dfrac{x^2}{2\sigma^2}\right)$",
              r"$f\left(x\right) = A\cos(kx)+\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\dfrac{x^2}{2\sigma^2}\right)$"]

    labels_fig2 = [r"$f$",
                   r"$\partial_x f$",
                   r"$\partial_x^2 f$",
                   r"$FD-\partial_x f$", r"$FD-\partial_x^2 f$"]

    xticks_fig1 = [[5, k_values[kmin_index[i, 0]],
                    k_values[kmin_index[i, 1]], 100] for i in range(3)]

    # grafica del error relativo
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey="all",
                           figsize=figsize, dpi=dpi)
    ax = ax.flatten()
    for i in range(ax.size):
        ax[i].set_title(titles[i], fontsize=fontsize)

        # dx
        ax[i].semilogy(k_values, errors[i][:, 0], color='C0',
                       marker=FDmarker[0], ls="None")
        # dxx
        ax[i].semilogy(k_values, errors[i][:, 1], color='C1',
                       marker=FDmarker[1], ls="None")

        xmin, xmax = ax[i].set_xlim()
        ymin, ymax = ax[i].set_ylim()

        ax[i].vlines(k_values[kmin_index[i, 0]], ymin,
                     alpha_error, linestyle="dotted", color="black")

        ax[i].vlines(k_values[kmin_index[i, 1]], ymin,
                     alpha_error, linestyle="dashdot", color="black")

        # #horizontal line
        ax[i].hlines(y=alpha_error, xmin=xmin, xmax=k_values[kmin_index[i].max(
        )], color="black", linestyle="dashed")

        ax[i].set_xticks(xticks_fig1[i])
    ax[0].set_ylabel("Relative error", fontsize=fontsize)
    # , bbox_to_anchor=(1.6, 1.01))
    ax[0].legend(labels_fig1, loc="upper right")
    ax[1].set_xlabel("Number of grid points per wavelength", fontsize=fontsize)
    # fig.text(0.53, 0.0, "Number of grid points per wavelength",
    #          va='center', ha='center', fontsize=12)
    fig.tight_layout()
    plt.savefig(savepath+"/relative_error_finite_difference_approximation.pdf")
    plt.show()
    plt.close()
#
    # ,figsize=(1024*px, 720*px)
    # #grafica de las aproximaciones
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex="col",
                           sharey="row", figsize=figsize, dpi=dpi)
    # ax = ax.flatten()
    for i in range(3):

        # 1st row images
        x, dx = np.linspace(domain[i][0], domain[i][1],
                            k_values[kmin_index[i, 0]], retstep=True)
        fx = fun[i](x)
        dxfx = dxfun[i](x)
        dxxfx = dxxfun[i](x)

        # derivative of the functions using the five points operator
        # 1st derivative
        d1_new = fp_operator(size=k_values[kmin_index[i, 0]], n=1, dx=dx)
        d1fx = d1_new@fx

        ax[0, i].set_title(titles[i], fontsize=fontsize)
        # derivada analitica
        ax[0, i].plot(x, (dxfx-dxfx.mean())/dxfx.std(ddof=1), color="C3", ls=fxls[1],
                      alpha=1, label=labels_fig2[1])
        # aproximacion de la derivada
        ax[0, i].plot(x, (d1fx-d1fx.mean())/d1fx.std(ddof=1), color="C0",
                      marker=FDmarker[0], label=labels_fig2[3], ls="None", alpha=0.7)

        ax[0, i].set_xlabel("{0} grid points per wavelength".format(
            k_values[kmin_index[i, 0]]), fontsize=fontsize)

        # 2nd derivative
        x, dx = np.linspace(domain[i][0], domain[i][1],
                            k_values[kmin_index[i, 1]], retstep=True)
        fx = fun[i](x)
        dxfx = dxfun[i](x)
        dxxfx = dxxfun[i](x)

        d2_new = fp_operator(size=k_values[kmin_index[i, 1]], n=2, dx=dx)
        d2fx = d2_new@fx

        # imagenes de la fila inferior
        # derivada analitica
        ax[1, i].plot(x, (dxxfx-dxxfx.mean())/dxxfx.std(ddof=-1), color="C4", ls=fxls[2],
                      alpha=1, label=labels_fig2[2])
        # aproximacion de la derivada
        ax[1, i].plot(x, (d2fx-d2fx.mean())/d2fx.std(ddof=1), color="C1",
                      marker=FDmarker[1], label=labels_fig2[4], ls="None", alpha=0.7)

        # labels y legendas
        ax[0, 0].legend(loc="upper right")
        ax[1, 0].legend(loc="upper right")
        ax[0, 0].set_ylabel('1st derivative')
        ax[1, 0].set_ylabel('2nd derivative')

        ax[1, i].set_xlabel("{0} grid points per wavelength".format(
            k_values[kmin_index[i, 1]]), fontsize=fontsize)

        ax[-1, i].set_xticks(xticks_fig2[i], labels=xticks_label_fig2[i])
    fig.tight_layout()
    plt.savefig(savepath+"/finite_difference_aproximation.pdf")
    plt.show()
    plt.close()

    pass
