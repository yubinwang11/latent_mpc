"""

Car model for this project

"""

import math
from math import cos, sin, tan

import matplotlib.pyplot as plt
import numpy as np

from common.angle import rot_mat_2d, pi_2_pi
from common.plot_utils import se2_to_trans_mat


class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, steer=0.0, a=0.0, steer_rate=0.0, vy=0.0, yaw_rate=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.steer = steer
        self.a = a
        self.vy = vy
        self.steer_rate = steer_rate
        self.yaw_rate = yaw_rate
        self.predelta = None
        self.speed = 0.0

class VehicleParams(object):
    Ts = 0.1  # [s] sampling time

    # Dynamic Parameters
    kf = -128916.0
    kr = -85944.0
    m = 1412.0  # mass (kg)
    lf = 1.06  # distance from CoG to front axle (m)
    lr = 1.85  # distance from CoG to rear axle (m)
    Iz = 1536.7  # moment of inertia (kg m^2)

    # Lk = (lf*kf) - (lr*kr)

    # Kinematic Parameters
    wheelbase = lf + lr  # distance between front and rear axles (m)
    max_steer_angle = 0.6  # maximum steering angle (rad)
    max_steer_rate = 0.6  # maximum steering rate (rad/s)
    max_acc = 1.5
    min_acc = -3.0
    max_v = 2.0  # maximum velocity (m/s)
    min_v = -1.0  # minimum velocity (m/s)
    # Collision Parameters
    width = 2.0  # width of vehicle (m)
    length = 4.8  # length of vehicle (m)
    front_axle_offset = 0.5  # distance from front end of vehicle to front axle (m)
    rear_axle_offset = length - front_axle_offset - wheelbase  # distance from rear end of vehicle to rear axle (m)

    # vehicle rectangle vertices
    vertice_x = [length / 2.0, length / 2.0, -length / 2.0, -length / 2.0, length / 2.0]
    vertice_y = [width / 2.0, -width / 2.0, -width / 2.0, width / 2.0, width / 2.0]

    bubble_dist = (rear_axle_offset+wheelbase+front_axle_offset)/2.0 - rear_axle_offset   # distance from rear to center of vehicle.
    # BUBBLE_DIST = (front_axle_offset + wheelbase - rear_axle_offset) / 2.0  # distance from rear to center of vehicle.
    bubble_radius = (rear_axle_offset+wheelbase+front_axle_offset)/2.0  # bubble radius


def check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
    for i_x, i_y, i_yaw in zip(x_list, y_list, yaw_list):
        cx = i_x + VehicleParams.bubble_dist * cos(i_yaw)
        cy = i_y + VehicleParams.bubble_dist * sin(i_yaw)

        ids = kd_tree.query_ball_point([cx, cy], VehicleParams.bubble_radius)

        if not ids:
            continue

        if not rectangle_check(i_x, i_y, i_yaw,
                               ox, oy):
            return False  # collision

    return True  # no collision


def rectangle_check(x, y, yaw, ox, oy):
    LF = VehicleParams.front_axle_offset + VehicleParams.wheelbase
    LB = VehicleParams.rear_axle_offset
    W = VehicleParams.width
    # transform obstacles to base link frame
    rot = rot_mat_2d(yaw)
    for iox, ioy in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]
        if not (rx > LF or rx < -LB or ry > W / 2.0 or ry < -W / 2.0):
            return False  # no collision

    return True  # collision


def generate_vehicle_vertices(x, y, yaw, base_link=False):
    L = VehicleParams.length
    W = VehicleParams.width
    b_to_f = VehicleParams.wheelbase + VehicleParams.front_axle_offset
    b_to_r = VehicleParams.rear_axle_offset

    vertice_x = []
    vertice_y = []
    if(base_link):
        vertice_x = [x + b_to_f*cos(yaw) - W/2*sin(yaw),
                     x + b_to_f*cos(yaw) + W/2*sin(yaw),
                     x - b_to_r*cos(yaw) + W/2*sin(yaw),
                     x - b_to_r*cos(yaw) - W/2*sin(yaw)]

        vertice_y = [y + b_to_f*sin(yaw) + W/2*cos(yaw),
                     y + b_to_f*sin(yaw) - W/2*cos(yaw),
                     y - b_to_r*sin(yaw) - W/2*cos(yaw),
                     y - b_to_r*sin(yaw) + W/2*cos(yaw)]
    else:
        vertice_x = [x + L/2*cos(yaw) - W/2*sin(yaw),
                     x + L/2*cos(yaw) + W/2*sin(yaw),
                     x - L/2*cos(yaw) + W/2*sin(yaw),
                     x - L/2*cos(yaw) - W/2*sin(yaw)]

        vertice_y = [y + L/2*sin(yaw) + W/2*cos(yaw),
                     y + L/2*sin(yaw) - W/2*cos(yaw),
                     y - L/2*sin(yaw) - W/2*cos(yaw),
                     y - L/2*sin(yaw) + W/2*cos(yaw)]

    V = np.vstack((vertice_x, vertice_y)).T

    return V


# Plot car with rear center at (x, y) and heading angle
def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    # Vehicle parameters
    LENGTH = VehicleParams.length  # [m]
    WIDTH = VehicleParams.width  # [m]
    BACKTOWHEEL = VehicleParams.rear_axle_offset  # [m]
    WHEEL_LEN = 0.3  # [m]
    WHEEL_WIDTH = 0.2  # [m]
    TREAD = 0.7  # [m]
    WB = VehicleParams.wheelbase  # [m]

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD,
                          -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")
    print(x, y)


def plot_car_by_center_pose(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
    plt.plot(x, y, "*r")
    trans_mat = se2_to_trans_mat(x, y, -yaw)
    trans_mat_inv = np.linalg.inv(trans_mat)
    rear_center_rel = np.array([-VehicleParams.lr, 0, 1]).reshape(3, 1)
    rear_center = trans_mat_inv @ rear_center_rel
    plot_car(rear_center[0], rear_center[1], yaw, steer, cabcolor, truckcolor)


def move(x, y, yaw, distance, steer, L=VehicleParams.wheelbase):
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw += pi_2_pi(distance * tan(steer) / L)  # distance/2

    return x, y, yaw


def main():
    x, y, yaw = 0., 0., 1.
    plt.axis('equal')
    # plot_car(x, y, yaw)
    plot_car_by_center_pose(x, y, yaw)
    plt.show()


if __name__ == '__main__':
    main()
