'''
Dimenssions convention
Length: Angstrom
Potential: volts
Projected Potential: in e = 14.4 volts \times Angstrom
'''

import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
import Specimen


class EleMSCP:
    def __init__(self, Cs, df = 0, beam_energy = 200, aperture = np.pi/2):

        self.beam_energy = beam_energy # energy in keV
        self.wave_len = 12.415/np.sqrt(self.beam_energy**2 \
            + 1021.998*self.beam_energy) # wave length in Angstrom
        self.Cs = Cs # spherical aberration in mm 
        self.df = df # defocus in Angstrom
        self.aperture = aperture
        self.specimen = None

    def aberration(self, k):
        # k = np.sqrt(kx*kx + ky*ky)
        chi = 0.5*np.pi*self.Cs*1e7*pow(self.wave_len, 3)*pow(k, 4) \
            - np.pi*self.df*self.wave_len*k*k
        return chi

    def __MTF(self, k):
        # k = np.sqrt(kx*kx + ky*ky)
        H = np.where(k<self.aperture/self.wave_len, np.exp(-1j*self.aberration(k)), 0)
        return H

    def set_beam_energy(self, energy):
        self.beam_energy = energy #keV

    def set_defocuse(self, df):
        self.df = df # Angstrom

    def set_aperture(self, a):
        self.aperture = a #rad

    def load_specimen(self, specimen):
        '''
        specimen: an instance of class Specimen
        '''
        self.specimen = specimen
    def plot_CTF(self, ax):
        kk = np.linspace(0, 0.6, 1000)
        CTF = np.sin(self.aberration(kk))
        ax.plot(kk, CTF)
        ax.set_title("CTF")
        ax.set_xlabel("1/Angstrom")


    def form_image(self):
        print("wave length = ", self.wave_len, "Angstrom")
        Nx = Ny = self.specimen.pix_number
        a = b = self.specimen.dimension
        ka = 0.5*Nx/a
        kb = 0.5*Ny/b
        kx, ky = np.mgrid[-ka:ka:Nx*1j, -kb:kb:Ny*1j]
        k = np.sqrt(kx*kx + ky*ky)

        # Fourier transforms the transmission function
        t = self.specimen.trans_func() # origin at center
        T = fftshift(fft2(t)) # origin at center
        # Contrast transfer function
        H = self.__MTF(k) # origin at center
        # Image wave in Fourier space, filtered by a circle bandwidth
        PSI = T*H # origin at center
        PSI = np.where(k<ka, PSI, 0)
        img_wave = ifft2(ifftshift(PSI)) # origin at top left
        img = abs(img_wave)**2
        return PSI, img

    def show_image(self, img, ax):
        ax.imshow(img, cmap="gray_r", interpolation="nearest")
        ax.set_title("image")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    def Thon_rings(self, img):
        IMG = fftshift(fft2(img))
        return abs(IMG)



if __name__=="__main__":
    fig, (ax1, ax2) = plt.subplots(ncols = 2)
    # set specimen
    atoms_list = [[6,0,-20,0], [14,0,-10,0], [29,0,0,0], [79,0,10,0], [92,0,20,0]]
    s = Specimen.SingleLayerAtoms(dimension=50, pix_number = 512)
    s.add_atoms(atoms_list)
    # s.show(ax1)

    # set EM
    beam_energy = 200
    Cs = 1.2 # mm
    em = EleMSCP(Cs = Cs, beam_energy=200)
    # Scherzer condition
    df = np.sqrt(1.5*em.Cs*1e7*em.wave_len)
    a = pow(6*em.wave_len/(em.Cs*1e7),0.25)
    print("Cs = ", Cs, "mm")
    print("aperture = ", a*1000, "mrad")
    print("defocus = ", df, "Angstrom")
    em.set_defocuse(df)
    em.set_aperture(a)
    em.load_specimen(s)
    PSI, img = em.form_image()
    Thon = em.Thon_rings(img)

    em.show_image(img, ax2)
    em.plot_CTF(ax1)
    plt.savefig("simulated_imgs/5atoms.png")
    plt.show()


