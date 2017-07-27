import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
import Atoms


class Specimen:
    def __init__(self, dimension = 50, pix_number = 512):
        '''
        dimension: side length of the specimen in unit Angstrom
        pix_number: pixel number along one side
        '''
        self.dimension = dimension
        self.pix_number = pix_number
        self.x, self.y = np.mgrid[-dimension/2:dimension/2:pix_number*1j, 
                        -dimension/2:dimension/2:pix_number*1j]
        self.proj_pot = np.zeros_like(self.x)

    def add_atoms(self):
        # Linear superposition for atoms:
        # self.proj_pot += np.sin(10*self.x+5*self.y)+np.sin(-10*self.x+5*self.y)
        C = Atoms.Atom(6, pos = [0, -20, 0])
        self.proj_pot += C.projected_potential(self.x, self.y).real
        Si = Atoms.Atom(14, pos = [0, -10, 0])
        self.proj_pot += Si.projected_potential(self.x, self.y).real
        Cu = Atoms.Atom(29, pos = [0, 0, 0])
        self.proj_pot += Cu.projected_potential(self.x, self.y).real
        Au = Atoms.Atom(79, pos = [0, 10, 0])
        self.proj_pot += Au.projected_potential(self.x, self.y).real
        U = Atoms.Atom(92, pos = [0, 20, 0])
        self.proj_pot += U.projected_potential(self.x, self.y).real

    def trans_func(self):
        return np.exp(1j*self.proj_pot)

    def show(self, ax):
        ax.imshow(self.proj_pot, cmap="gray_r")


class EM:
    def __init__(self, Cs, df = 0, beam_energy = 200, aperture = np.pi/2):

        self.beam_energy = beam_energy # energy in keV
        self.wave_len = 12.415/np.sqrt(self.beam_energy**2 \
            + 1021.998*self.beam_energy) # wave length in Angstrom
        self.Cs = Cs # spherical aberration in mm 
        self.df = df # defocus in Angstrom
        self.aperture = aperture
        self.specimen = Specimen()

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

    def Thon_rings(self, img):
        IMG = fftshift(fft2(img))
        return abs(IMG)





if __name__=="__main__":
    fig, (ax1, ax2) = plt.subplots(ncols = 2)
    # set specimen
    s = Specimen(dimension=50)
    s.add_atoms()
    s.show(ax1)
    # set EM
    beam_energy = 200

    Cs = 1.2 # mm
    em = EM(Cs = Cs, beam_energy=200)
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

    # ax2.imshow(abs(PSI), cmap="gray_r",interpolation="nearest")
    ax2.imshow(img, cmap="gray_r", interpolation="nearest")
    # ax4.imshow(Thon, cmap="gray_r", interpolation="nearest")


    ax1.set_title("proj_pot")
    ax2.set_title("image")
    # ax3.set_title("image")
    # ax4.set_title("Thon rings")
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    # ax3.xaxis.set_visible(False)
    # ax3.yaxis.set_visible(False)
    # ax4.xaxis.set_visible(False)
    # ax4.yaxis.set_visible(False)
    plt.savefig("simulated_imgs/5atoms.png")
    plt.show()


