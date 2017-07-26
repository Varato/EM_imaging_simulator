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

	def __init__(self, Z):
		self.Z = int(Z)
		if not 1<=self.Z<=103:
			print("Wrong atom Z number!")
			quit()
		pm = atom_pot_pms[Z-1]
		self.chiq = pm[0]
		self.a = pm[1:6:2]
		self.b = pm[2:7:2]
		self.c = pm[7:12:2]
		self.d = pm[8:13:2]

	def projected_potential(self, x, y):
		r = np.sqrt(x*x + y*y)
		sum1 = sum([self.a[i]*kn(0,2*np.pi*r*np.sqrt(self.b[i])) for i in range(3)])
		sum2 = sum([self.c[i]*np.exp(-np.pi*np.pi*r*r/self.d[i])/self.d[i] for i in range(3)])
		return 2*np.pi*np.pi*a0*(2*sum1 + sum2)*e


r = np.linspace(0.1, 0.5)


x,y = np.mgrid[-2:2:512j, -2:2:512j]
Au = Atom(79)
proj_pot = Au.projected_potential(x,y).real
# # print(proj_pot)
plt.imshow(proj_pot, cmap="gray")
plt.show()











