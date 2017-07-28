import numpy as np
import matplotlib.pyplot as plt
import EMsims.EM as EM
import EMsims.Specimen as Specimen


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = [9,8])
# set EM
beam_energy = 200
Cs = 1.2 # mm
em = EM.EleMSCP(Cs = Cs, beam_energy=200.)
# Scherzer condition
df = np.sqrt(1.5*em.Cs*1e7*em.wave_len)
a = pow(6*em.wave_len/(em.Cs*1e7),0.25)
print("Cs = ", Cs, "mm")
print("aperture = ", a*1000, "mrad")
print("defocus = ", df, "Angstrom")
em.set_defocuse(df)
em.set_aperture(a)

# set specimen
atoms_list = [[6,25,5,0], [14,25,15,0], [29,25,25,0], [79,25,35,0], [92,25,45,0]]
s = Specimen.SingleLayerAtoms(dimension=50, pix_number = 513)
s.add_atoms(atoms_list)
s.plot_proj_pot(ax1, along = int(s.pix_number/2)+1)
em.load_specimen(s)
PSI, img = em.form_image()
Thon = em.Thon_rings(img)

em.show_image(img, fig, ax2)
em.plot_CTF(ax3)

img_intensity = img[257]
xxx = np.linspace(0, 50, 513)
ax4.plot(xxx, img_intensity)
ax4.set_title("image intensity along centers of atoms")
ax4.set_xlabel("Angstrom")

plt.savefig("simulated_imgs/5atoms.png")
plt.show()
