'''
Dimenssions convention
Length: Angstrom
Potential: volts
Projected Potential: in e = 14.4 volts \times Angstrom
'''

import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt

m0cq = 510.9989461 #keV

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

    def trans_func(self):
        interact_pm = (14.39964/1975.9)*(m0cq + \
            self.beam_energy)/np.sqrt(self.beam_energy*(2*m0cq + self.beam_energy)) # 1/e
        return np.exp(1j*interact_pm*self.specimen.proj_pot)

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
        t = self.trans_func() # origin at center
        T = fftshift(fft2(t)) # origin at center
        # Contrast transfer function
        H = self.__MTF(k) # origin at center
        # Image wave in Fourier space, filtered by a circle bandwidth
        PSI = T*H # origin at center
        PSI = np.where(k<ka, PSI, 0)
        img_wave = ifft2(ifftshift(PSI)) # origin at top left
        img = abs(img_wave)**2
        return PSI, img

    def show_image(self, img,fig, ax):
        img = ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.set_title("image")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.colorbar(img, ax=ax)
    def Thon_rings(self, img):
        IMG = fftshift(fft2(img))
        return abs(IMG)





