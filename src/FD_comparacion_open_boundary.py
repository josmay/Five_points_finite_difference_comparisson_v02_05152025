'''
This script made the comparison between differentes finite differences methods: five-points, back-difference, central-difference.
'''

import os

import matplotlib.pyplot as plt
import numpy as np

from FD_solucion_onda_5p import FD_5p
from FD_solucion_onda_BD import FD_BD
from FD_solucion_onda_CD import FD_CD
from FD_solucion_onda_FD import FD_FD
from pictures_setup import dpi, figsize, fontsize

if __name__ == "__main__":

    norma1 = []
    norma2 = []
    norma3 = []
    norma4 = []

    x_vals = range(5, 50, 5)
    for lambda_nodes in x_vals:
        
        z_norm1 = FD_5p(lambda_nodes)
        # z_norm2 = FD_FD(lambda_nodes)
        z_norm3 = FD_BD(lambda_nodes)
        z_norm4 = FD_CD(lambda_nodes)

        norma1.append(z_norm1)
        # norma2.append(z_norm2)
        norma3.append(z_norm3)
        norma4.append(z_norm4)

    std1 = [np.std(i, ddof=-1) for i in norma1]
    # std2 = [np.std(i, ddof=-1) for i in norma2]
    std3 = [np.std(i, ddof=-1) for i in norma3]
    std4 = [np.std(i, ddof=-1) for i in norma4]

    
    # Create the figure and axis for the plot
    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)

    # Plotting each data series on the left y-axis (primary axis)
    line1, = ax1.semilogy(x_vals, std1, 'C0*', label="FD_5p")  # Red markers for std1
    ax1.semilogy(x_vals, std1)                          # Black line for std1
    line3, = ax1.semilogy(x_vals, std3, 'C1^', label="FD_BD")  # Cyan triangle markers for std3
    ax1.semilogy(x_vals, std3)                          # Black line for std3
    line4, = ax1.semilogy(x_vals, std4, 'C2s', label="FD_CD")  # Green square markers for std4
    ax1.semilogy(x_vals, std4)   

    # Adding title and labels to the left axis
    ax1.set_title("Standard deviation of wave discretization at the boundary",fontsize=fontsize)
    ax1.set_xlabel("Number of grid point per wavelength",fontsize=fontsize)
    ax1.set_ylabel("Standard deviation",fontsize=fontsize)

    # # Combine legends
    # # lines = [line1, line2, line3, line4]
    lines = [line1, line3,line4]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower right")
    plt.savefig(savepath+"/finite_difference_comparison.pdf")
    # Show the plot
    plt.show()
    plt.close()
