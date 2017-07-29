import numpy as np
import matplotlib.pyplot as plt
import EMsims.EM as EM
import EMsims.Specimen as Specimen


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = [9,8])
# silicon parameter

a0 = 3.84
b0 = 5.43

# set EM
Cs = 1.3 # mm
em = EM.EleMSCP(Cs = Cs, beam_energy=400.)
# Scherzer condition
df = np.sqrt(1.5*em.Cs*1e7*em.wave_len)
a = pow(6*em.wave_len/(em.Cs*1e7),0.25)
print("Cs = ", Cs, "mm")
print("aperture = ", a*1000, "mrad")
print("defocus = ", df, "Angstrom")
em.set_defocuse(df)
em.set_aperture(a)

# set specimen
y0 = [yy*b0 for yy in list(map(lambda n: n/2 if n%2==0 else 0.25+(n-1)/2, range(11)))]
y1 = [yy + 0.5*b0 for yy in y0][:-1]
atoms_list = [[14 ,0.5*x*a0, y, 0.] for x in range(0,15,2) for y in y0] \
                + [[14, 0.5*x*a0, y, 0.] for x in range(1,14,2) for y in y1]
s = Specimen.SingleLayerAtoms(dx = 7*a0, dy = 5*b0, nx = 128, ny = 128)
s.add_atoms(atoms_list)
# s.plot_projpot_horiz(ax1, int(128/7))
s.show(ax1, cmap = "nipy_spectral")
# form image
em.load_specimen(s)
PSI, img = em.form_image()
Thon = em.Thon_rings(img)
em.show_image(img, fig, ax2)
em.plot_CTF(ax3)

# further analysis
img_intensity = img[64]
xxx = np.linspace(0, 7*a0, 128)
ax4.plot(xxx, img_intensity)
ax4.set_title("image intensity along centers of atoms")
ax4.set_xlabel("Angstrom")

plt.savefig("simulated_imgs/silicon110.png")
plt.show()
