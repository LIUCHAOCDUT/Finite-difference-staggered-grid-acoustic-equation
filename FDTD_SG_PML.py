# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:22:06 2023

@author: 34655
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 15:41:42 2023

@author: 34655
"""

'''
仅供个人学习使用！！


For personal learning use only!!

'''
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation

from threading import Thread
def standardizer(data):
    std_d = data.flatten()
    std_data = (data - np.mean(data)) / np.std(std_d)
    std_data[std_data > 1] = 1
    std_data[std_data < -1] = -1
    return std_data

def FDTD_SG_PML(velocity, NX, NZ, NT, SX, SZ):
    
    
    
    
    
    
    nx = NX  # Number of grid nodes in x-direction.
    nz = NZ  # Number of grid nodes in z-direction.
    npmlz = 30  # Number of grid nodes in top and bottom side of PML absorbing boundary.
    npmlx = 30  # Number of grid nodes in left and right side of PML absorbing boundary.
    sx = SX  # Grid node number of source position in x-direction.
    sz = SZ  # Grid node number of source position in z-direction.
    dx = 10  # Grid node interval in x-direction; Unit: m.
    dz = 10  # Grid node interval in z-direction; Unit: m.
    nt = NT  # Number of time nodes for wave calculating.
    dt = 1e-3  # Time node interval; Unit: s.
    nppw = 12  # Node point number per wavelength for dominant frequency of Ricker wavelet source.
    ampl = 1.0  # Amplitude of source wavelet.
    xrcvr = np.arange(0, nx, 3)  # Grid node number in x-direction of receiver position on ground.
    nodr = 6  # Half of the order number for spatial difference.

    # Determine the difference coefficients
    B = np.array([1] + [0] * (nodr - 1)).astype(np.double)
    A = np.zeros((nodr, nodr)).astype(np.double)
    for i in range(nodr):
        A[i, :] = (np.arange(1, 2 * nodr, 2).astype(np.double)**(2 * (i+1) - 1))
    C = np.linalg.solve(A, B)

    # Model and source
    Nz = nz + 2 * npmlz
    Nx = nx + 2 * npmlx
    vp = np.ones((Nz, Nx),dtype=np.double)  # Initialize vp with ones.
    
    vp=np.pad(velocity,(npmlz,npmlx),'edge')
    
    
    
    
    #地表反射
    vp[:npmlz,:]=1

    

    # Calculate rho
    rho = (1.66 * (vp / 1000)**0.261 * 1000)

    # Calculate source wavelet parameters
    f0 = 20
    t0 = 1 / f0
    t = dt * np.arange(1, nt + 1)
    src = (1 - 2 * (np.pi * f0 * (t - t0))**2) * np.exp(- (np.pi * f0 * (t - t0))**2)

    # Perfectly matched layer absorbing factor
    dpml0z = 3 * np.max(vp) / dz * (8 / 15 - 3 / 100 * npmlz + 1 / 1500 * npmlz**2)
    dpmlz = np.zeros((Nz, Nx))
    dpmlz[:npmlz, :] = (dpml0z * ((np.arange(npmlz, 0, -1) / npmlz)**2)).reshape(-1, 1) @ np.ones((1, Nx))
    dpmlz[npmlz + nz:, :] = dpmlz[npmlz-1::-1, :]

    dpml0x = 3 * np.max(vp) / dx * (8 / 15 - 3 / 100 * npmlx + 1 / 1500 * npmlx**2)
    dpmlx = np.zeros((Nz, Nx))
    dpmlx[:, :npmlx] = np.ones((Nz, 1)) @ (dpml0x * ((np.arange(npmlx, 0, -1) / npmlx)**2)).reshape(1, -1)
    dpmlx[:, npmlx + nx:] = dpmlx[:, npmlx-1::-1]

    # Wavefield calculating
    rho1 = rho
    rho2 = rho

    Coeffi1 = (2 - dt * dpmlx) / (2 + dt * dpmlx)
    Coeffi2 = (2 - dt * dpmlz) / (2 + dt * dpmlz)
    Coeffi3 = 1.0 / rho1 / dx * (2.0 * dt / (2.0 + dt * dpmlx))
    Coeffi4 = 1.0 / rho2 / dz * (2.0 * dt / (2.0 + dt * dpmlz))
    Coeffi5 = rho * (vp**2.0) / dx * (2.0 * dt / (2.0 + dt * dpmlx))
    Coeffi6 = rho * (vp**2.0) / dz * (2.0 * dt / (2.0 + dt * dpmlz))

    NZ = Nz + 2 * nodr
    NX = Nx + 2 * nodr
    Znodes = np.arange(nodr, NZ - nodr)                     #正演区域索引值
    Xnodes = np.arange(nodr, NX - nodr)                     #正演区索引值
    znodes = np.arange(nodr + npmlz, nodr + npmlz + nz)     #扩展区域索引值
    xnodes = np.arange(nodr + npmlx, nodr + npmlx + nx)     #扩展区域索引值
    nsrcz = nodr + npmlz + sz
    nsrcx = nodr + npmlx + sx

    Ut = np.empty((NZ, NX))
    Uz = np.zeros((NZ, NX))
    Ux = np.zeros((NZ, NX))
    Vz = np.zeros((NZ, NX))
    Vx = np.zeros((NZ, NX))
    Psum = np.full((Nz, Nx),0).astype(np.double)

    U = np.full((nz, nx, nt),0).astype(np.double)
    for it in range(nt):
            if it % 500 == 0:
                print(f'The calculating time node is: it = {it}')
            
            Ux[nsrcz-1, nsrcx-1] += ampl * src[it] / 2.0
            Uz[nsrcz-1, nsrcx-1] += ampl * src[it] / 2.0
            Ut[:, :] = Ux[:, :] + Uz[:, :]
            U[:, :, it] = Ut[znodes[:, np.newaxis], xnodes]
            
            Psum=0
            
            for i in range(nodr):
                Psum += C[i] * (Ut[Znodes[:, np.newaxis], Xnodes + (i+1)] - Ut[Znodes[:, np.newaxis], Xnodes + 1 - (i+1)])
            
            Vx[Znodes[:, np.newaxis], Xnodes] = Coeffi1 * Vx[Znodes[:, np.newaxis], Xnodes] - Coeffi3 * Psum
            Psum=0
            for i in range(nodr):
                Psum += C[i] * (Ut[(Znodes + (i+1))[:, np.newaxis], Xnodes] - Ut[(Znodes + 1 - (i+1))[:, np.newaxis], Xnodes])
            
            Vz[Znodes[:, np.newaxis], Xnodes] = Coeffi2 * Vz[Znodes[:, np.newaxis], Xnodes] - Coeffi4 * Psum
            Psum=0
            
            for i in range(nodr):
                Psum += C[i] * (Vx[Znodes[:, np.newaxis], Xnodes - 1 + (i+1)] - Vx[Znodes[:, np.newaxis], Xnodes - (i+1)])
            
            Ux[Znodes[:, np.newaxis], Xnodes] = Coeffi1 * Ux[Znodes[:, np.newaxis], Xnodes] - Coeffi5 * Psum
            Psum=0
            
            for i in range(nodr):
                Psum += C[i] * (Vz[(Znodes - 1 + (i+1))[:, np.newaxis], Xnodes] - Vz[(Znodes - (i+1))[:, np.newaxis], Xnodes])
            
            Uz[Znodes[:, np.newaxis], Xnodes] = Coeffi2 * Uz[Znodes[:, np.newaxis], Xnodes] - Coeffi6 * Psum
    
    return U




if __name__ == '__main__':

    # Example usage
    nx = 301
    nz = 201
    v = np.ones((nz, nx)) * 2000
    v[100:, :] = 4000
    v[200:, :] = 5000
    velocity = v
    sz = 1
    nt = 2001
    
    
    
    
    for i in range(1):
        sx = 121
        SYN = np.zeros((nx, nz, nt))
        SYN = FDTD_SG_PML(velocity, nx, nz, nt, sx, sz)
        qw = np.transpose(SYN, (2, 1, 0))
    seismogram4 = standardizer(qw[:, :, 0])
        
        
    # 创建一个图形对象
    fig, ax = plt.subplots()
    
    # 定义动画更新函数
    def update(frame):
        ax.clear()  # 清空当前图形
        slice_2d = SYN[:,:,frame]
        ax.imshow(slice_2d, cmap='viridis',aspect='auto')  # 使用合适的颜色映射
        ax.set_title(f'2D Slice {frame+1}')
    
    # 创建动画对象
    ani = FuncAnimation(fig, update, frames=SYN.shape[2], interval=50)  # interval 控制帧之间的时间间隔（毫秒）
    
    # 显示动画
    
    plt.figure()
    plt.imshow(seismogram4,aspect='auto')
    plt.show()
    # Rest of the code...



