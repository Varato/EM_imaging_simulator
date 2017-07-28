'''
Dimenssions convention
Length: Angstrom
Potential: volts
Projected Potential: in e = 14.4 volts \times Angstrom
hc = 12.415 keV Angstrom
hbarc = 1.9759 keV Angstrom
'''

import numpy as np
from scipy.special import kn
import matplotlib.pyplot as plt

a0 = 0.529177 #Angstrom
e = 14.39964 #volts Angstrom
atom_pot_pms = np.loadtxt("external_data/atom_pot_pm.csv", delimiter=",")
r_min = 1e-10


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

        sum1 = sum([self.a[i]*kn(0, 2*np.pi*r_min*np.sqrt(self.b[i])) for i in range(3)])
        sum2 = sum([self.c[i]*np.exp(-np.pi*np.pi*r_min*r_min/self.d[i])/self.d[i] for i in range(3)])
        v_max = 2*np.pi*np.pi*a0*(2*sum1 + sum2)
        xx = x - self.proj_pos[0]
        yy = y - self.proj_pos[1]
        r = np.sqrt(xx*xx + yy*yy)
        sum1 = sum([self.a[i]*kn(0,2*np.pi*r*np.sqrt(self.b[i])) for i in range(3)])
        sum2 = sum([self.c[i]*np.exp(-np.pi*np.pi*r*r/self.d[i])/self.d[i] for i in range(3)])
        return np.where(r>r_min, 2*np.pi*np.pi*a0*(2*sum1 + sum2), -r*r+v_max+r_min*r_min)

    def plot_potential(self, ax):
        x,y = np.mgrid[-2:2:512j, -2:2:512j]
        proj_pot = self.projected_potential(x, y).real
        ax.imshow(proj_pot, cmap="gray_r")


class SingleLayerAtoms:

    def __init__(self, dimension = 50, pix_number = 512):
        '''
        dimension: side length of the specimen in unit Angstrom
        pix_number: pixel number along one side
        '''
        self.dimension = dimension
        self.pix_number = pix_number
        self.x, self.y = np.mgrid[0:dimension:pix_number*1j, 0:dimension:pix_number*1j]
        self.proj_pot = np.zeros_like(self.x)

    def add_atoms(self, atoms_list):
        '''
        atoms_list: Z, x, y, z
        '''
        # Linear superposition for atoms:
        for atom in atoms_list:
            self.proj_pot += Atom(atom[0], pos = atom[1:]).projected_potential(self.x, self.y).real


    def show(self, ax):
        ax.imshow(self.proj_pot, cmap="gray_r")
        ax.set_title("proj_pot")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    def plot_proj_pot(self, ax, along):
        ax.plot(np.linspace(0, self.dimension, self.pix_number), self.proj_pot[along])
        ax.set_title("proj_pot")
        ax.set_ylabel("Angstrom")
        ax.set_ylabel("in unit e (14.4 volts$\cdot$Angstrom)")



