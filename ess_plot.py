import fileinput
import time
from datetime import datetime
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy.signal import periodogram
from psutil import cpu_percent
import py2exe

# subplot globals
fig1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8,8))
fig2, (ax4, ax5, ax6) = plt.subplots(3, figsize=(8,8))
#fig, ax7 = plt.subplots(1)     # psutil

# set up lines for animation
line1, = ax1.plot([], [])
line2, = ax2.plot([], [])
line3, = ax3.plot([], [])
line4, = ax4.plot([], [])
line5, = ax5.plot([], [])
line6, = ax6.plot([], [])
#line7, = ax7.plot([], [])     # psutil

# create line and x+y data arrays for convenience
ch1_lines = [line1, line2, line3]
ch2_lines = [line4, line5, line6]
x1, x2, y1, y2, y3 = [], [], [], [], []

# initializes subplot figures
def initFigures():
    
    for i, figure in enumerate([fig1, fig2]):
        figure.tight_layout(pad=4)
        figure.suptitle('Channel %d Data Analysis' % int(i+1), y=0.98, fontsize=12, fontweight='semibold')
        figure.subplots_adjust(top=0.9)

# initializes individual plots in each subplot figure
def initSubplots():
    
    ax1.set_title('ch1 Raw Output')
    ax1.set(xlabel='Samples')
    ax1.set(ylabel='Magnitude (uint16)')
    ax1.set_xlim(0, 500)
    ax1.set_ylim(0, 65536)
    
    ax2.set_title('ch1 Fourier Transform')
    ax2.set(xlabel='Frequency (Hz)')
    ax2.set(ylabel='FFT Magnitude')
    ax2.set_xlim(0, 500)
    
    ax3.set_title('ch1 Power Spectral Density')
    ax3.set(xlabel='Frequency (Hz)')
    ax3.set(ylabel='PSD Magnitude')
    ax3.set_xlim(0, 500)

    ax4.set_title('ch2 Raw Output')
    ax4.set(xlabel='Samples')
    ax4.set(ylabel='Magnitude (uint16)')
    ax4.set_xlim(0, 500)
    ax4.set_ylim(0, 65536)
    
    ax5.set_title('ch2 Fourier Transform')
    ax5.set(xlabel='Frequency (Hz)')
    ax5.set(ylabel='FFT Magnitude')
    ax5.set_xlim(0,500)
    
    ax6.set_title('ch2 Power Spectral Density')
    ax6.set(xlabel='Frequency (Hz)')
    ax6.set(ylabel='PSD Magnitude')
    ax6.set_xlim(0,500)

def initLines():
    
    for chn in [ch1_lines, ch2_lines]:
        for line in chn:
            line.set_data([], [])
    
    return line,

def plotChannels():

    # plot channel 1 analysis
    ax1.plot(ch1data)
    ax2.plot(f ,np.abs(ch1_fft[0:515]))
    ax3.plot(f, np.abs(ch1_psd[1][0:515]))

    # plot channel 2 analysis
    ax4.plot(ch2data)
    ax5.plot(f ,np.abs(ch2_fft[0:515]))
    ax6.plot(f, np.abs(ch2_psd[1][0:515]))


def animatech1(i):

    # update the data
    x1.append(x[i])
    x2.append(f[i])
    y1.append(ch1data[i])
    y2.append(abs(ch1_fft[i]))
    y3.append(abs(ch1_psd[1][i]))

    # check for correct axis size,  resize if necessary
    for ax in [ax1, ax2, ax3]:
        
        # get current limits
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # resize raw data x-axis if necessary (frequency axis is fixed)
        if ax == ax1 and x1[i] >= xmax:
            ax.set_xlim(xmin, 2*xmax)
            ax.figure.canvas.draw()
        
        # resize FFT y-axis if necessary
        if ax == ax2 and y2[i] >= ymax:
            ax.set_ylim(ymin, 2*ymax)
            ax.figure.canvas.draw()

        # resize PSD y-axis if necessary
        if ax == ax3 and y3[i] >= ymax:
            ax.set_ylim(ymin, y3)
            ax.figure.canvas.draw()

    # update data in each line object
    ch1_lines[0].set_data(x1, y1)
    ch1_lines[1].set_data(x2, y2)
    ch1_lines[2].set_data(x2, y3)
    
    return ch1_lines

def animatech2(i):

    # update the data
    x1.append(x[i])
    x2.append(f[i])
    y1.append(ch2data[i])
    y2.append(abs(ch2_fft[i]))
    y3.append(abs(ch2_psd[1][i]))

    # check for correct axis size, resize if necessary
    for ax in [ax4, ax5, ax6]:
        
        # get current limits
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # resize raw data x-axis if necessary (frequency axis is fixed)
        if ax == ax4 and x1[i] >= xmax:
            ax.set_xlim(xmin, 2*xmax)
            ax.figure.canvas.draw()
        
        # resize FFT y-axis if necessary
        if ax == ax5 and y2[i] >= ymax:
            ax.set_ylim(ymin, 2*ymax)
            ax.figure.canvas.draw()

        # resize PSD y-axis if necessary
        if ax == ax6 and y3[i] >= ymax:
            ax.set_ylim(ymin, 2*ymax)
            ax.figure.canvas.draw()

    # update data in each line object
    ch2_lines[0].set_data(x1, y1)
    ch2_lines[1].set_data(x2, y2)
    ch2_lines[2].set_data(x2, y3)
    
    return ch2_lines

def animatepsutil(i):
    xt.append(datetime.now())
    yt.append(cpu_percent())
    line7.set_data(xt, yt)
    fig.gca().relim()
    fig.gca().autoscale_view()
    return line7

#################################### MAIN ####################################

# initialize channel data lists
ch1data = []
ch2data = []

for line in fileinput.input():
    
    # concatenate and convert to int
    ch1 = int('0x' + line[12:17].replace(' ','') , 16)
    ch2 = int('0x' + line[18:23].replace(' ','') , 16)

    ch1data.append(ch1)
    ch2data.append(ch2)

# remove the mean value to scale the signal for FFT / PSD calculation
voltageRange = 5

ch1_scaled = []
ch2_scaled = []

for i, val in enumerate(ch1data):
    ch1_scaled.append( ((ch1data[i]/65535)*voltageRange*2)-voltageRange )
    ch2_scaled.append( ((ch2data[i]/65535)*voltageRange*2)-voltageRange )

# define useful constants
Fs = 1000                                               # sampling frequency
T = 1/Fs                                                # period

# calculate fft for each channel
fftSize = 1028                                          # fft algo performs optimally when size is a power of 2
ch1_fft = fft(ch1_scaled, fftSize)
ch2_fft = fft(ch2_scaled, fftSize)

# calculate psd for each channel
ch1_psd = periodogram(ch1_scaled, Fs, nfft=fftSize)
ch2_psd = periodogram(ch2_scaled, Fs, nfft=fftSize)

# get some unit step arrays for plotting purposes
f = []
x = []

# populate unit step arrays
for i in range( int(fftSize/2) + 1 ):
    f.append(Fs * (i / fftSize))        # normalized
    x.append(i)

# plot the data
initFigures()
initSubplots()
plotChannels()

# create animations
#ch1_animation = FuncAnimation(fig1, animatech1, frames=None, init_func=initLines, blit=True, interval=5, repeat=True)
#ch2_animation = FuncAnimation(fig2, animatech2, frames=None, init_func=initLines, blit=True, interval=5, repeat=True)

plt.show(block=True)

#xt, yt = [], []

#animation = FuncAnimation(fig, animatepsutil, interval=1)

#fig.show()
#plt.show()


