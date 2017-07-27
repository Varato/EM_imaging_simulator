'''
Dimenssion convention
Length: Angstrom
Potential: volts
Projected Potential: in e = 14.4 volts Angstrom
'''

import numpy as np
from scipy.special import kn
import matplotlib.pyplot as plt

a0 = 0.529177 #Angstrom
e = 14.39964 #volts Angstrom
atom_pot_pms = np.loadtxt("atom_pot_pm.csv", delimiter=",")


class Atom:

    def __init__(self, Z, pos = [0,0,0]):
        self.Z = int(Z)
        if not 1<=self.Z<=103:
            print("Wrong atom Z number!")
            quit()
        self.pos = np.array(pos)
        self.proj_pos = pos[:2]
        pm = atom_pot_pms[Z-1]
        self.chiq = pm[0]
        self.a = pm[1:6:2]
        self.b = pm[2:7:2]
        self.c = pm[7:12:2]
        self.d = pm[8:13:2]

    def projected_potential(self, x, y):
        xx = x - self.proj_pos[0]
        yy = y - self.proj_pos[1]
        r = np.sqrt(xx*xx + yy*yy)
        sum1 = sum([self.a[i]*kn(0,2*np.pi*r*np.sqrt(self.b[i])) for i in range(3)])
        sum2 = sum([self.c[i]*np.exp(-np.pi*np.pi*r*r/self.d[i])/self.d[i] for i in range(3)])
        return 2*np.pi*np.pi*a0*(2*sum1 + sum2)*e

    def plot_potential(self, ax):
        x,y = np.mgrid[-2:2:512j, -2:2:512j]
        proj_pot = self.projected_potential(x, y).real
        ax.imshow(proj_pot, cmap="gray")







