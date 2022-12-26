
from math import sin, cos, pi, tan
import matplotlib.pyplot as plt
import numpy as np


def se2_to_trans_mat(x, y, yaw):
    return np.array([[cos(yaw), -sin(yaw), x],
                        [sin(yaw), cos(yaw), y],
                        [0, 0, 1]])


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """Plot arrow."""
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(x, y, length * cos(yaw), length * sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)