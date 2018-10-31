# -*- coding: utf-8 -*-
"""
@author: Sameer Kesava

written in anaconda spyder
"""

""" Derivation of Intrinsic Series Resistance by extricating the Sheet Resistance
of transparent conductive oxide (ITO in this case) for a approximated rectangular 
solar cell by 1D numerical integration along its width while keeping length constant"""

#%%
"""Solar cell derived parameters from fitting I-V data 
to the diode equation"""

Rseries = 2.52248 #Ohms-cm2
Rshunt = 579.563 #Ohms-cm2
Jph = 10.6305*10**(-3) #A/cm2
n = 3.14219 #ideality factor
Jo = 4.33538*10**(-8)  #A/cm2
Rsheet = 20 #Ohms/sq
Vth = 25.6825 * 10**(-3) #Thermal voltage-eV
#Voc = 1.0013 #Volt
w = 0.3 #measured cell width
l = 0.3 #measured cell length
#%%
""" importing packages"""
import math
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#from scipy.integrate import odeint
from scipy.special import lambertw
#import sys
from scipy import optimize
import time
#%%
def Currentdensity(v, R):
    """ Calculates Current density J as a function of applied bias, V, and
    series resistance using Lambert-W function. Returns complex value"""   
    return -v/(R+Rshunt)\
    -lambertw(R*Jo*Rshunt\
    *math.exp(Rshunt*(R*(Jph+Jo)+v)\
    /(n*Vth*(R+Rshunt)))/(n*Vth*(R+Rshunt)))\
    *n*Vth/R + Rshunt*(Jph+Jo)/(R+Rshunt)
#%%
""" Solving for Open-Circuit Voltage"""
def func2(x):
    return Jph-Jo*(2.71828**(x/(n*Vth))-1)\
    -x/(Rshunt)

solVoc = optimize.root(func2, 0.98, method = 'lm')
Voc=solVoc.x[0]
#%%
Vinlist = np.arange(0.01, Voc+0.01, 0.01)
#Initializing input voltages at x = 0

#%%
"""Dividing the cell of width w into multiple cells of width dx"""
dx = 0.0001 

#Integration step

Num = int(round(w/dx,0))    
#Number of data points for integration

widthlist = np.arange(0, w+dx, dx)
#%%
def fun(x,*argv):
    """Function representing diode-equation with input voltage and cell
    intrinsic series resistance as arguments"""
    return x-Jph+Jo*(math.exp((argv[0]+argv[1]*x)/(n*Vth))-1)\
    +(argv[0]+argv[1]*x)/(Rshunt)

def Jx(v,i,r):
    """ Returns Current density J by solving the diode equation while passing
    input voltage and cell intrinsic series resistance as arguments to the 
    diode-equation function"""
    sol=optimize.root(fun, i, args = (v,r), method = 'lm')
    return sol.x[0]


#%%
def VIlist(VIinput, Rsheet, Rseries):
    """ Inputting voltage and current values at x=0 along cell width.
    For a given sheet resistance of electrode and series resistance of active layer.
    Returns the Voltage and Current at x=w"""
   
    Vlist = np.array([VIinput[0]]) 
    Ilist = np.array([VIinput[1]])
    for i in range(Num):   
        Inext = Ilist[i]+ Jx(Vlist[i],Ilist[i], Rseries)*dx*l
        Vnext = Vlist[i] - Rsheet*dx/l*Inext
        Ilist = np.append(Ilist, Inext)
        Vlist = np.append(Vlist, Vnext)
       
    
    return [Vlist[-1], Ilist[-1]]
#%%
""" Root Mean Squared Error is calculated from the difference between the
J obtained from the diode-equation fit to the measured data using one
coupled series resistance and J from the fit to the fit data where sheet 
and series resistance are decoupled. The residual is normalized by 10^-3"""

from joblib import Parallel, delayed


def errfunc(rs):
    err = 0 #Initializing current density residual error for each input voltage
    
    Voutlist = np.array([])
    #Initializing input voltage array at x=width 
    Ioutlist = np.array([])
    #Initializing output current array at x=width 
    Jlist = np.array([])
    
    
    for j in range(len(Vinlist)):
        
        data = VIlist([Vinlist[j],0], Rsheet, rs)
        Voutlist = np.append(Voutlist, data[0])
        Ioutlist = np.append(Ioutlist, data[1])
        
        Jlist = np.append(Jlist, Jx(Voutlist[j], Ioutlist[j]/(w*l), Rseries))
        
        err = err + (Ioutlist[j]*1000/(l*w) - Jlist[j]*1000)**2
        #Sum of errors at each input voltages        
        
    return [rs, np.sqrt(err/len(Vinlist)*1000)]
    
#%%
"""Parallel Processing"""
start_time = time.time()
n_jobs = 3
listofMSE = Parallel(n_jobs = n_jobs)(delayed(errfunc)(i) for i in np.arange(0, Rseries*2, 0.1))    

end_time = time.time()
#%%
print('Time taken: {:.2f} min'.format ((end_time-start_time)/60))
"""With n_jobs = 4, time taken was 25 min to scan from 0 to 2*Rseries.
Without parallel processing, time taken was 61 min and with n_jobs = 8, took 18 min """
#%%
"""Creating a list of MSEs obtained from using a set of Rsheet above"""
MSError = np.empty((0,2))
for i in listofMSE:
    MSError = np.append(MSError, [i], axis = 0)

    
#%%
fig1, ax1 = plt.subplots()
ax1.plot(MSError[:,0], MSError[:,1] , 'm-', lw=2,\
label = r'Sheet R: %d $\Omega$/sq' %(Rsheet))
ax1.legend(loc ='upper left', prop={'size':10})
plt.title('Determination of Intrinsic Series Resistance of the Solar Cell')
plt.xlabel(r'Solar Cell Intrinsic Series Resistance ($\Omega$-cm$^2$)')
plt.ylabel('Root Mean Squared Error')
plt.show()
#%%
fig1.savefig('RMSerror_vs_Rintrinsic.png', format = 'png', dpi=100)
#%%
np.savetxt("RMerror_vs_Rintrinsic.csv", MSError, delimiter=",", header = "Rintrinsic, RMS",\
comments = "")
#%%
index = np.argmin(MSError[:,1])
#index of minimum MSE

Rintrinsic = MSError[:,0][index]
#Rintrinsic of the cell
#%%

"""Deriving IV data for cells with Rseries and Rintrinic as the series resistances"""
Voutlist_Rs = np.array([]) #Rseries from fit to experimental data
Ioutlist_Rs = np.array([])

Voutlist_Ri = np.array([]) #Rintrinsic from including Rsheet in the fit
Ioutlist_Ri = np.array([])

#Can also parallelize this for faster output
for i in Vinlist:
    temp_ri = VIlist([i,0], Rsheet, Rintrinsic)
    temp_rs = VIlist([i,0], Rsheet, Rseries)
    Voutlist_Ri = np.append(Voutlist_Ri, temp_ri[0])
    Ioutlist_Ri = np.append(Ioutlist_Ri, temp_ri[1])
    Voutlist_Rs = np.append(Voutlist_Rs, temp_rs[0])
    Ioutlist_Rs = np.append(Ioutlist_Rs, temp_rs[1])
    
    
Jlist = np.array([])
for i in Vinlist:
    Jlist = np.append(Jlist, Currentdensity(i, Rseries).real)
   
#%%
"""Plotting together for comparison. The cell with Rsheet added to Rintrinsic show replicate 
the working solar cell given by the Diode equation"""
fig2, ax2 = plt.subplots()
ax2.plot(Voutlist_Ri, Ioutlist_Ri*1000/(w*l), 'b--', lw=1,\
label = r'R Intrinsic = %0.1f $\Omega$-cm$^2$, R sheet = %d $\Omega$/sq' %(Rintrinsic, Rsheet))
ax2.plot(Voutlist_Rs, Ioutlist_Rs*1000/(w*l), 'r--', lw=1, \
         label = r'R Series = %0.1f $\Omega$-cm$^2$, R sheet = %d $\Omega$/sq' %(Rseries, Rsheet))
ax2.plot(Vinlist, Jlist*1000, 'g--', lw=1, label = r'Diode equation, \
Rseries: %0.1f $\Omega$-cm$^2$'%Rseries)
ax2.legend(loc ='lower left', prop={'size':8})
ax2.axis([0, np.max(Vinlist), 0, np.max(Jlist)*1000])
plt.title(r'I-V comparison while including Sheet Resistance: %d $\Omega$/sq' %(Rsheet))
plt.xlabel('Voltage (V) ')
plt.ylabel(r'Current density (mA/cm$^2$)')
plt.show()
#%%
fig2.savefig('IV_RintrinsicDetermination_v3.png', format = 'png', dpi=100)    
    