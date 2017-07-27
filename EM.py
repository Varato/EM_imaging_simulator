import numpy as np
from scipy.special import kn
import matplotlib.pyplot as plt
import Atoms


class Specimen:
    def __init__(self, dimensions = 4, pix_number = 512):
        x, y = np.mgrid[-dimensions/2:dimensions/2:pix_number*1j, 
                        -dimensions/2:dimensions/2:pix_number*1j]
        self.proj_pot = np.zeros_like(x)

        # Linear superposition for atoms:
        Au = Atoms.Atom(79, pos = [1,1,0])
        self.proj_pot += Au.projected_potential(x, y).real
        Cu = Atoms.Atom(29, pos = [-1,1,0])
        self.proj_pot += Cu.projected_potential(x, y).real

    def trans_func(self):
        return np.exp(1j*self.proj_pot)






class EM:
    def __init__(self, Cs, df = 0):
        self.Cs = Cs
        self.df = df



if __name__=="__main__":
    fig, ax = plt.subplots(1)
    s = Specimen()
    ax.imshow(s.proj_pot, cmap="gray")
    plt.show()



