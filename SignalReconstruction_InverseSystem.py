#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 18:48:29 2022

@author: feng
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.interpolate import interp1d
from InverseSystem import *


# Image Output Setting
plt.rcParams['figure.dpi'] = 300

# Balloon Parameters
v_up = 2


# RH and Temperature Data
# height_sv = np.array([4, 200, 800, 1100, 1500])
# Temp_sv = np.array([20, 17, 12, 11, 10])
# RH_sv = np.array([65, 63, 80, 55, 60])
# height_interp = np.linspace(4, 1500, int((1500-4)/2 + 1))


# Environment Data
height_sv = np.array([4, 20, 80, 110, 150])
Temp_sv = np.array([20, 17, 12, 11, 10])
RH_sv = np.array([65, 63, 80, 55, 60])
height_interp = np.linspace(4, 150, int((150-4)/2 + 1))

# Cubic Spline Interpolation
fun_Temp = interp1d(height_sv, Temp_sv, kind = 'cubic')
Temp_interp = fun_Temp(height_interp)

fun_RH = interp1d(height_sv, RH_sv, kind = 'cubic')
RH_interp = fun_RH(height_interp)


# Wagner Equation
A = -7.76451
B = 1.45838
C = -2.7758
D = -1.23303
Pc = 221200 # [hPa] Critical Pressure
Tc = 647.3 # [K] Critical Temperature

t = Temp_interp
x = 1 - (t + 273.15) / Tc
et = Pc * np.exp( (A * x + B * x**1.5 + C * x**3 + D * x**6) / (1 - x) )
at = 217 * et / (t + 273.15)
WVM_interp = RH_interp / 100 * at  


# Plot of Environment Data
plt.Figure()
plt.suptitle("cubic spline interpolation")
plt.subplot(1,3,1)
plt.plot(Temp_interp, height_interp, label="Temp(interpolation)")
plt.scatter(Temp_sv, height_sv, c='r')
plt.xlabel("Temperature / centi")
plt.ylabel("Height / m")

plt.subplot(1,3,2)
plt.plot(RH_interp, height_interp, label="RH(interpolation)")
plt.scatter(RH_sv, height_sv, c='r')
plt.xlabel("RH / %")
plt.yticks(())

plt.subplot(1,3,3)
plt.plot(WVM_interp, height_interp, label="RH(interpolation)")
plt.xlabel("WVM / g")
plt.yticks(())
plt.show()


# Simulation of Balloon fly in velocity = v_up
sens = sim_sensor(RH_interp, height_interp)
fft_plot(sens, len(sens), Ts=1)
step_plot(sens, len(sens), 1,  sens[0])



# Add white noise to humidity signal
snr = 60  # Signal-to-noise ratio
white_noise = produce_whitenoise(sens, snr)
# Plot 
sens_wn = sens+white_noise
# fft_plot(sens_wn, len(sens_wn), Ts=1)
# step_plot(sens_wn, len(sens_wn), 1, sens_wn[0])
inv = fir_plot(sens_wn, len(sens_wn), 1, sens_wn[0])


env = RH_interp
env_start = np.ones(10) * RH_interp[0]
env_stop = np.ones(10) * RH_interp[-1]
env = np.append(env_start, env)
env = np.append(env, env_stop)


plt.Figure()
plt.plot(env, label='env')
plt.plot(inv, label='inv')
plt.plot(sens_wn[1000-10:-(1000-10)], label='sens')
plt.xlabel('Time / s')
plt.ylabel('Humidity / %')
plt.legend()
plt.title("Signal Construction by Inverse System")
plt.show()