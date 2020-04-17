# -*- coding: utf-8 -*-
##import callibration
'''Enter filename with data depth energy cross section'''
SIMNRA = 'C:/Users/kolesar/Documents/cv/51_Schönaich bei Böblingen/04_depth_profiel_analysis/Ga in Si - en_stop_range.dat'

'''Enter filename with measured data'''
measurements = 'C:/Users/kolesar/Documents/cv/51_Schönaich bei Böblingen/04_depth_profiel_analysis/RBS measurements for depth profile/1_keV/6_1keV_0073_random'

'''Enter fluence [at/cm2]'''
Fluence = 1e15
'''Enter energy per channel'''
##filename1 = 'callibration.dat'
##dataFile = open(filename1, 'r')
##for line in dataFile:
##    Det_Offset1, Det_Calib1 = line.split()
##dataFile.close()
Det_Calib = 1.75#float(Det_Calib1)
Det_Offset = 1.088#float(Det_Offset1)
print ('Det_Calib=', Det_Calib)
print ('Det_Offset=', Det_Offset)
'''Enter solid angle'''
solid_angle = 3.362
'''Enter accumulated charge in micro Coulomb'''
charge = 9.65976177094
'''Enter slope of energy depth'''
k11 = -1.0
'''Enter intercept of energy depth'''
q11 = 1380.0
'''Enter slope of cross section depth'''
k22 = 1.0/1250
'''Enter intercept of cross section depth'''
q22 = 1.80

'''Enter names of output files'''
results3 = 'C:/Users/kolesar/Documents/cv/51_Schönaich bei Böblingen/04_depth_profiel_analysis/results_FZR/channel_counts_Ga_in_Si/0073_2_Si_Wafer/1_keV/6_1keV_0073.dat'
results4 = 'C:/Users/kolesar/Documents/cv/51_Schönaich bei Böblingen/04_depth_profiel_analysis/results_FZR/depth_area_density_Ga_in_Si/0073_2_Si_Wafer/1_keV/6_1keV_0073.dat'
results5 = 'C:/Users/kolesar/Documents/cv/51_Schönaich bei Böblingen/04_depth_profiel_analysis/results_FZR/depth_norm_vol_density_Ga_in_Si/0073_2_Si_Wafer/1_keV/6_1keV_0073.dat'
figure1 = 'C:/Users/kolesar/Documents/cv/51_Schönaich bei Böblingen/04_depth_profiel_analysis/results_FZR/multiplefigures/0073_2_Si_Wafer/1_keV/6_1keV_0073.png'
figure2 = 'C:/Users/kolesar/Documents/cv/51_Schönaich bei Böblingen/04_depth_profiel_analysis/results_FZR/figures/0073_2_Si_Wafer/1_keV/6_1keV_0073.png'

'''Enter constant for multiple figure'''
c=0.1
'''Enter range of channels with implanted ions'''
first_channel = 750
last_channel = 790
'''Enter gaussain fitting parameters'''
amplitude  =2.1e14
peak_position  = 5.0
'''Depth range'''
x4_min = -50
x4_max =250
y4_min = 0
y4_max = 5e14

import pylab as plt
import numpy as np

from depth_reader import get_table1

depth1, energy1, sigma1 = get_table1(SIMNRA)
g1 = depth1[::-1]
h1 = energy1[::-1]
h2 = sigma1[::-1]

'''fitting procedure of energy vs depth'''

x1 = np.linspace(g1[0], g1[-1], len(g1))
    
def f1(x1, k11, q11):
    '''Fit function y = f(x,p) with parameters p = (k, q).'''
    return k11*x1 + q11
'''depth energy data determined from SIMNRA'''
y1 = h1
y1_ = f1(x1, k11, q11)
from scipy.optimize import curve_fit
popt1, pcov1 = curve_fit(f1, x1, y1)
k1, q1 = popt1
yfitted1 = f1(x1, *popt1)

'''fitting procedure of sigma vs depth'''

def f2(x1, k22, q22):
    '''Fit function y = f(x,p) with parameters p = (k, q).'''
    return k22*x1 + q22
'''sigma energy data determined from SIMNRA'''
y2 = h2
y2_ = f2(x1, k22, q22 )
popt2, pcov2 = curve_fit(f2, x1, y2)
k2, q2 = popt2
yfitted2 = f2(x1, *popt2)

from spectrum_reader import get_table2
from energy_reader import energy


channel1, counts1 = get_table2(measurements)
x3 = channel1[::-1]
y3 = counts1[::-1]
x3_min =700
x3_max = 800
y3_min = 0
y3_max = 500
k_dept = k1
q_dept = q1
k_sigm = k2
q_sigm = q2


energy, depth, sigma, area_density, volume_density, normalized_vol_density = energy(channel1, Det_Calib, Det_Offset, q_dept, k_dept, k_sigm, q_sigm, counts1, solid_angle, charge, Fluence)
'''writing data depth area_density'''
outfile3 = open(results3,'w')
for i in range(len(x3)):
    x_input = x3[i]
    y_input = y3[i]
    outfile3.write('%12.8e %12.8e\n' % (x_input, y_input))
outfile3.close()

###########################################################################################################################################################################################

x4  = depth
y4 = area_density

Max_Noise_Level = 2.36203e14
Min_Noise_Level = 1.46719e14
Mean_Noise_Level = (Max_Noise_Level+Min_Noise_Level)/2
Fluence1 = 0
for j1 in range(first_channel,last_channel ):
    Fluence1 = Fluence1 + y4[j1]
Fluence1 = Fluence1 - Mean_Noise_Level
print ('Fluence1=', Fluence1)
'''writing data depth area_density'''
outfile4 = open(results4,'w')
for i in range(len(x4)):
    x_input = x4[i]
    y_input = y4[i]
    outfile4.write('%12.8e %12.8e\n' % (x_input, y_input))
outfile4.close()
###############################################################
from area_density_reader import get_table4
depth4, area_density4 = get_table4(results4)

g4 =[]
h4 = []
mean  =0
for j in range(len(depth4)):
    g4.append(float(depth4[j]))
    h4.append(float(area_density4[j]))
    mean = mean + depth4[j]*area_density4[j]
difference = depth4[j] - depth4[j-1]

sigma_gauss = 0
for i in range(len(depth4)):
    sigma_gauss = sigma_gauss + (area_density4[i]*(depth4[i]-mean)**2)
    
'''fitting procedure of area density vs depth'''
x4_ = np.linspace(depth4[0],depth4[-1],len(depth4))
##x4_= np.linspace(depth4[
##y4_ = area_density4

##def f4(x4_, amplitude, peak_position, sigma_gauss):
##    '''Fit function y = f(x, p) with parameters p = (amplitude, peak_position, sigma_gauss).'''
##    return amplitude*np.exp(-(x4_-peak_position)**2)/(2*sigma_gauss**2)
##
##y4_=f4(x4_, amplitude, peak_position, sigma_gauss)
##popt4, pcov4 = curve_fit(f4, x4_, y4_)
##amplitude, peak_position, sigma_gauss = popt4
##yfitted4 = f4(x4_, *popt4)

##mean = 0
##sigma_gauss = 0
##for j in range(len(depth)):
##    x4.append(float(depth[j]))
##    y4.append(float(area_density[j]))
##    bracket = depth[j]*area_density[j]
##    mean = mean + bracket
##for i in range(mean):
##    sigma_gauss = sigma_gauss + (area_density[j]*(depth[j]-mean)**2)
##print sigma_gauss
##'''calculated data'''

################################################################################

x5 = depth
y5 = normalized_vol_density
x5_min = 0
x5_max = 100
y5_min = 0
y5_max = 3e5
'''writing data depth normalized volume density'''
outfile5 = open(results5,'w')
for i in range(len(x4)):
    x_input = x5[i]
    y_input = y5[i]
    outfile5.write('%12.8e %12.8e\n' % (x_input, y_input))
outfile5.close()

# Simple data to display in various forms

##plt.close('all')
# row and column sharing
plt.figure(figsize = (10,6))
plt.subplot(2, 2, 1)
plt.title('E_Det vs. dedpth')
plt.plot(x1,y1,'o', label ='data')
plt.plot(x1, yfitted1, '-', label = "fit $f1(x1)$")
plt.xlabel('Depth [nm]')
plt.ylabel('E [keV]')
plt.annotate('$f1(x1) = k1*x1+ q1$', xy=(380, 1250))
plt.annotate("$k1 =%g$" %(k1), xy =(430, 1200))
plt.annotate("$q1 =%g$" %(q1), xy =(430, 1150))
##plt.xticks(())
##plt.yticks(())
plt.grid()
plt.legend(prop={'size':12})
##plt.text(1300, 200, 'E_Det vs. depth', ha='center', va='center',
##        size=20, alpha=.5)

plt.subplot(2, 2, 2)
plt.title('sigma vs. depth')
plt.plot(x1,y2,'o', label ='data')
plt.plot(x1, yfitted2, '-', label = "fit $f2(x2)$")
plt.xlabel('Depth [nm]')
plt.ylabel('Sigma [b/sr]')

plt.annotate('$f2(x2) = k2*x2+ q2$', xy=(20, 2.06-c))
plt.annotate("$k2 =%g$" %(k2), xy =(20, 2.0-c))
plt.annotate("$q2 =%g$" %(q2), xy =(20, 1.94-c))
##plt.xticks(())
##plt.yticks(())
plt.grid()
plt.legend(loc='upper left',prop={'size':12})
##plt.text(0.5, 0.5, 'subplot(2,2,2)', ha='center', va='center',
##        size=20, alpha=.5)

plt.subplot(2, 2, 3)
plt.title('raw data')
plt.plot(x3,y3, label='spectrum')
plt.xlim(x3_min, x3_max)
plt.ylim(y3_min, y3_max)
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.annotate("$Det.Calib =%g$" %(Det_Calib), xy =(600, 6000))
plt.annotate("$Det.Offset =%g$" %(Det_Offset), xy=(600,4000))
plt.annotate("$charge =%g$" %(charge), xy=(600, 2000))

##plt.xticks(())
##plt.yticks(())
plt.grid()
plt.legend(prop={'size':12})
##plt.text(600, 12500.0, 'raw data', ha='center', va='center',
##        size=15, alpha=.5)

plt.subplot(2, 2, 4)
plt.title('area density vs. depth')
plt.plot(x4,y4, 'o', label = 'data')
##plt.plot(x4_,y4_, 'r-', label = 'data')
##plt.plot(x4, yfitted4, '-', label = "fit $f4(x4,amplitude,peak_position,sigma_gauss)$")
plt.annotate("$Fluence1 =%g$" %(Fluence1), xy =(100, 3.5e14))
##plt.annotate("$sigma_gauss =%g$" %(sigma_gauss), xy =(100, 2.5e14))
plt.xlim(x4_min, x4_max)
plt.ylim(y4_min, y4_max)
plt.xlabel('Depth [nm]')
plt.ylabel('Area density [at/cm2]')
##plt.xticks(())
##plt.yticks(())
plt.grid()
plt.legend(prop={'size':12})
##plt.text(0.5, 0.5, 'subplot(2,2,4)', ha='center', va='center',
##        size=20, alpha=.5)
plt.tight_layout()
plt.savefig(figure1)


plt.figure(figsize = (10,6))
plt.title('norm. volume density vs. depth')
plt.plot(x5,y5, 'ro', label = 'data')
plt.xlim(x5_min, x5_max)
plt.ylim(y5_min,y5_max)
plt.xlabel('Depth [nm]')
plt.ylabel('norm. vol. density ([at/cm3]/[at/cm2])')
plt.grid()
plt.legend()
plt.savefig(figure2)
