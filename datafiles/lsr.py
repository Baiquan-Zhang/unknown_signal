import os
import sys
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values

def view_data_segments(xs, ys):
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()

def fold_shuffle(k):
    assert k <= 20
    l = np.array(range(20))
    np.random.shuffle(l)
    return np.array_split(l, k)

def linear_regression(xs, ys):
    x_v = xs.reshape(np.size(xs), 1)
    ones = np.ones(xs.shape).reshape(np.size(xs), 1)
    x = np.column_stack((ones, x_v))
    y = ys.reshape(np.size(ys), 1)
    result = np.linalg.solve(x.T.dot(x), x.T.dot(y))
    a, b = result[:, 0]

    y_c = a + b*xs
    recon_err = ((ys - y_c) ** 2).sum()

    return a, b, recon_err

def polynomial_regression(xs, ys, max_order):
    
    cal_deg = 0
    cal_cof_list = np.zeros(max_order)
    cal_recon_err = sys.maxsize
    deg_recon_err_list = np.zeros(max_order)
    #up to polynomials of degree ten
    for degree in range(max_order):
        #return a randomly shuffled array of range(20)
        #5-fold for now
        r_list = fold_shuffle(5)
        deg_val_err = sys.maxsize
        deg_cof_list = np.zeros(degree+2)
        deg_recon_err = sys.maxsize
        #5-fold, cross-val for each group
        for i in range(5):
            xs_v = np.zeros(4)
            ys_v = np.zeros(4)
            for j in range(4):
                xs_v[j] = xs[r_list[i][j]]
                ys_v[j] = ys[r_list[i][j]]
            xs_t = np.delete(xs, r_list[i])
            ys_t = np.delete(ys, r_list[i])

            deg_count_t = 1
            x = np.ones(xs_t.shape).reshape(np.size(xs_t), 1)
            while deg_count_t <= degree+1:
                x_foo = xs_t.reshape(np.size(xs_t), 1) ** deg_count_t
                x = np.column_stack((x, x_foo))
                deg_count_t += 1
            y = ys_t.reshape(np.size(ys_t), 1)
            result = np.linalg.solve(x.T.dot(x), x.T.dot(y))
            foo_cof_list = result[:,0]

            y_c = np.zeros(xs_v.shape)
            for deg_count_v in range(degree+2):
                y_c += foo_cof_list[deg_count_v]*(xs_v**deg_count_v)
            foo_val_err = ((ys_v - y_c) ** 2).mean()
            
            #check that this validation error is smaller than current smallest
            #if it is, update val_err, cof_list and recon_err  
            if foo_val_err < deg_val_err:
                deg_val_err = foo_val_err
                deg_cof_list = foo_cof_list
                ys_recon = np.zeros(ys.shape)
                for deg_count_recon in range (degree+2):
                    ys_recon += foo_cof_list[deg_count_recon]*(xs**deg_count_recon)
                deg_recon_err = ((ys - ys_recon) ** 2).sum()
                
        #check the reconstruction error of this degree is smaller than the current smallest
        #if it is, update deg, val_err, cof_list and recn_err
        if deg_recon_err < cal_recon_err:
            cal_recon_err = deg_recon_err
            cal_deg = degree + 1
            cal_cof_list = deg_cof_list
        
        deg_recon_err_list[degree] = deg_recon_err
    
    return cal_deg, cal_cof_list, cal_recon_err, deg_recon_err_list

def unknown_regression(xs, ys):
    
    g_recon_err = np.zeros(3)
    g_a = np.zeros(3)
    g_b = np.zeros(3)
    
    #exponential 
    g_recon_err[0], g_a[0], g_b[0] = calculate_unknown(xs, ys, 1)
    #sin function
    g_recon_err[1], g_a[1], g_b[1] = calculate_unknown(xs, ys, 2)
    #cos function
    g_recon_err[2], g_a[2], g_b[2] = calculate_unknown(xs, ys, 3)
    
    reg_type = g_recon_err.argmin()
    return reg_type+1, g_a[reg_type], g_b[reg_type], g_recon_err[reg_type]

def find_x(xs, reg_type):
    return {
        0: xs,
        1: np.exp(xs),
        2: np.sin(xs),
        3: np.cos(xs)
    }[reg_type]

def calculate_unknown(xs, ys, reg_type):
    x = find_x(xs, reg_type)
    un_a, un_b, un_recon_err = linear_regression(x, ys)
    return un_recon_err, un_a, un_b

        
def segment_reconstruction_error(xs, ys, ax):
    seg_recon_err = 0
    l_a, l_b, l_recon_err = linear_regression(xs, ys)
    p_deg, p_cof_list, p_recon_err, _ = polynomial_regression(xs, ys, 3)
    un_type, un_a, un_b, un_recon_err = unknown_regression(xs, ys)
    thre_l_recon_err = l_recon_err*(0.80)
    #polynomial type
    if p_recon_err < un_recon_err:
        if thre_l_recon_err <= p_recon_err:
            seg_recon_err = l_recon_err
            add_graph(xs, ys, ax, np.array([l_a, l_b]), 0)
        else:
            seg_recon_err = p_recon_err
            add_graph(xs, ys, ax, p_cof_list, 0)
    #unknown type
    elif un_recon_err <= p_recon_err:
        if thre_l_recon_err <= un_recon_err:
            seg_recon_err = l_recon_err
            add_graph(xs, ys, ax, np.array([l_a, l_b]), 0)
        else:
            seg_recon_err = un_recon_err
            add_graph(xs, ys, ax, np.array([un_a, un_b]), un_type)
    
    return seg_recon_err
    
def add_graph(xs, ys, ax, cof_list, reg_type):
    graph_xs = np.linspace(xs.min(), xs.max(), 100)
    cal_xs = find_x(graph_xs, reg_type)
    graph_ys = 0
    for i, cof in enumerate(cof_list):
        graph_ys += cof*(cal_xs**i)
    ax.scatter(xs, ys, c='b')
    ax.plot(graph_xs, graph_ys, c='r')
    
filename = sys.argv[1]
xs, ys = load_points_from_file(filename)
#parser = argparse.ArgumentParser()
#parser.add_argument('--plot', action='store_true')
#args = parser.parse_args()

assert len(xs) == len(ys)
assert len(xs) % 20 == 0
len_data = len(xs)
num_segments = len_data // 20
xs_list = np.array_split(xs, num_segments)
ys_list = np.array_split(ys, num_segments)
total_recon_err = 0
fig, ax = plt.subplots()
for i in range(num_segments):
    total_recon_err += segment_reconstruction_error(xs_list[i], ys_list[i], ax)

print(total_recon_err)

#if args.plot:
    plt.show()
