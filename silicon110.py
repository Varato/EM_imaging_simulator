import numpy as np
import matplotlib.pyplot as plt
import EMsims.EM as EM
import EMsims.Specimen as Specimen


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = [9,8])
# silicon parameter

a0 = 3.84
b0 = 5.43

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
y0 = [yy*b0 for yy in list(map(lambda n: n/2 if n%2==0 else 0.25+(n-1)/2, range(11)))]
y1 = [yy+0.5*b0 for yy in y0]
atoms_list = [[14 ,x*a0, y, 0] for x in range(0,7,2) for y in y0] \
                + [[14, x*a0, y, 0] for x in range(1,8,2) for y in y1[:-1]]
s = Specimen.SingleLayerAtoms(dimension=55, pix_number = 513)
s.add_atoms(atoms_list)
s.show(ax1)
plt.show()
quit()
# form image
em.load_specimen(s)
PSI, img = em.form_image()
Thon = em.Thon_rings(img)
em.show_image(img, fig, ax2)
em.plot_CTF(ax3)

# further analysis
img_intensity = img[257]
xxx = np.linspace(0, 50, 513)
ax4.plot(xxx, img_intensity)
ax4.set_title("image intensity along centers of atoms")
ax4.set_xlabel("Angstrom")

plt.savefig("simulated_imgs/5atoms.png")
plt.show()
