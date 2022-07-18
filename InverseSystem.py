"""
Created on Thu Mar 10 09:19:04 2022

@author: Feng
"""


"""
Introduction:
    This program performs a signal reconstruction of the sensors on the balloon
    based on Inverse System and FIR filter
    
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.interpolate import interp1d


def fft_plot(humidity, Num, Ts):
# Frequency domain analysis diagram of output humidity signal
    fft_y = np.fft.fft(humidity)
    x = np.arange(Num)
    abs_y = np.abs(fft_y)
    # angle_y = np.angle(fft_y)


    norm_abs_y = abs_y / Num
    half_x = x[range(int(Num/2))]
    norm_half_abs_y = norm_abs_y[range(int(Num/2))]

    half_frq = half_x * Ts / Num

    # plt.figure()
    # plt.subplot( 2, 1, 1)
    # plt.plot(half_frq[1:], norm_half_abs_y[1:])
    # plt.ylabel("Amplitude / dB")
    # plt.subplot( 2, 1, 2)
    # plt.plot(half_frq[-50:], norm_half_abs_y[-50:])
    # plt.suptitle("Amplitude Spectrum")
    # plt.xlabel("Frequency / Hz")
    # plt.ylabel("Amplitude / dB")
    # plt.grid()
    # plt.show()

def step_plot(sens, Num, Ts, init_humi):
# Output the reconstructed signal and perform frequency domain analysis
    inv=np.zeros(Num)#Initialization
    inv[0] = init_humi
    c = 1 / float("inf")
    c = 0
    # plt.Figure()
    # plt.plot(sens, label="Sensor")
    # plt.title('Sensor with noise')
    # plt.xlabel('Time / s')
    # plt.ylabel('Humidity / %')
    # plt.legend()
    # plt.grid()
    # plt.show()

    #Simulation
    for k in range(Num-1):
        inv[k+1] = ((16 + Ts)*sens[k+1] + (Ts - 16)*sens[k] - (Ts - 2*c)*inv[k]) / (2*c + Ts)

    #Plot the Simulation Results
    # plt.Figure()
    # plt.plot(inv, label="InverseSystem")
    # plt.plot(sens, label="Sensor with noise")
    # #Formatting the appearance of the plot
    # plt.title('Inverse System')
    # plt.xlabel('Time / s')
    # plt.ylabel('Humidity / %')
    # plt.legend(loc='upper left')
    # plt.grid()
    # plt.show()
    
    
    # fft_plot(inv, len(inv), Ts)
    return inv

def fir_plot(sens, Num ,Ts, humi_init):
# FIR low-pass filtering is performed on the original signal, and the 
# reconstructed signal is output and analyzed in the frequency domain.
    sample_rate = 1

    # The Nyquist rate of the signal.
    nyq_rate = sample_rate / 2.0

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 0.1 Hz transition width.
    width = 0.01/nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = signal.kaiserord(ripple_db, width)

    # The cutoff frequency of the filter.
    cutoff_hz = 0.1

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = signal.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

    # group delay of filter 
    w, gd = signal.group_delay((taps, 1.0))
    delay = int(np.mean(gd))

    # transient of filter (order of filter)
    transi = len(taps)

    # Use lfilter to filter x with the FIR filter.
    sens_filt = signal.lfilter(taps, 1.0, sens)

    #------------------------------------------------
    # # Frequency Response of FIR filter.
    #------------------------------------------------
    w, h = signal.freqz(taps, 1.0)
    # plt.Figure()
    # plt.subplot(2,1,1)
    # plt.title('Digital filter frequency response')
    # plt.plot(w[:]/(2*np.pi), 20 * np.log10(abs(h[:])), 'b')
    # plt.ylabel('Amplitude [dB]', color='b')


    # plt.subplot(2,1,2)
    # angles = np.unwrap(np.angle(h))
    # plt.plot(w/(2*np.pi), angles, 'g')
    # plt.ylabel('Angle (radians)', color='g')
    # plt.xlabel('Frequency [Hz]')
    # plt.grid()
    # plt.axis('tight')
    # plt.show()
    
    # Plot Raw Data and Filtered Data
    # plt.Figure()
    # plt.title("FIR filter")
    # plt.plot(sens, label="Humidity_RawData")
    # plt.plot(sens_filt, label="Humidity_filtered")
    # plt.legend()
    # plt.show()
    
    
    sens = sens_filt[1+delay+transi:]
    sens = sens_filt[1000+delay-10 : -(1000-delay-10)]
    inv = step_plot(sens, len(sens), Ts, humi_init)

    return inv


def sim_sensor(humi, height):
    # In order to deal with group delay and transient of filter
    # Append some samples before and after the signal (sample length is 
    # determined by FIR filter performance)
    env = np.append(np.ones(1000) * humi[0], humi)
    env = np.append(env, np.ones(1000) * humi[-1])
    height = np.append(np.ones(1000) * height[0], height)
    height = np.append(height, np.ones(1000) * height[-1])
    
    l = len(env) + 2
    sens = np.zeros(l)
    N = l
    Ts = 1
    for k in range(N-2):
        if k==0:
            sens[k-1] = humi[0]
            env[k-1] = humi[0]
        sens[k] = (Ts*env[k] + Ts*env[k-1] - (Ts-16)*sens[k-1])/(16 + Ts)

    env = env[:-3]
    sens = sens[:-3]
    
    plt.Figure()
    plt.title("Sensor Simulation with Time")
    plt.plot(env, label='humidity')
    plt.plot(sens, label='sensor')
    plt.xlabel('Time / s')
    plt.ylabel('RH / %')
    plt.legend()
    plt.show()
    
    plt.Figure()
    plt.title("Sensor Simulation with Time")
    plt.plot(env, label='humidity')
    plt.plot(sens, label='sensor')
    plt.xlim(0, 1800)
    plt.xlabel('Time / s')
    plt.ylabel('RH / %')
    plt.legend()
    plt.show()
    
    
    # plt.Figure()
    # plt.title("Sensor Simulation with Height")
    # plt.plot(env, height[:-3], label='humidity')
    # plt.plot(sens[:-2], height, label='sensor')
    # plt.xlabel('Time / s')
    # plt.ylabel('RH / %')
    return sens


def produce_whitenoise(x, snr):
    len_x = len(x)
    Ps = np.sum(np.power(x, 2)) / len_x
    Pn = Ps / (np.power(10, snr/10))
    noise = np.random.randn(len_x) * np.sqrt(Pn)
    
    ### Plot ###
    plt.Figure()
    plt.plot(x, label="White Noise", c='b')
    plt.xlabel("Time / s")
    plt.ylabel("Humidity / %")
    plt.legend()
    plt.title("white_noise")
    plt.show()
    
    return noise

