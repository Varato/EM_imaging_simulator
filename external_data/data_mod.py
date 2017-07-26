import numpy as np

f = open("kirkland_atom_pot.dat")
raw_data = f.readlines()
f.close()

data = np.zeros([103, 12+1])

for Z in range(103):
	line1 = raw_data[Z*4]
	line2 = raw_data[Z*4+1]
	line3 = raw_data[Z*4+2]
	line4 = raw_data[Z*4+3]
	chiq = float(line1.split("=")[2].strip())
	data[Z][0] = chiq
	data[Z][1:5] = list(map(float, line2.strip().split(" ")))
	data[Z][5:9] = list(map(float, line3.strip().split(" ")))
	data[Z][9:13] = list(map(float, line4.strip().split(" ")))
np.savetxt("atom_pot_pm.csv", data, delimiter=',')