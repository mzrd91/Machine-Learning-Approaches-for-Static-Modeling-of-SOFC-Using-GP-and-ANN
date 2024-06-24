# Parameters used in the SOFC stack
number_of_stacks = 1
temperature_values = [873, 973, 1073, 1173, 1273] # List of temperature values in Kelvin
E0 = 1.18 # V
N0 = 384
u = 0.8
Kr = 0.993e-3 # mol/(s A)
Kr_cell = Kr / N0
KH2 = 0.843 # mol/(s atm)
KH2O = 0.281 # mol/(s atm)
KO2 = 2.52 # mol/(s atm)
r = 0.126 # U
rcell = r / N0
rHO = 1.145
i0_den = 20 # mA/cm2
ilimit_den = 900 # mA/cm2
A = 1000 # cm2
n = 2

F = 96485

alpha = 0.5
R = 0.0821 # atm/(mol K)
N = 20

PH2 = 1.265
PO2 = 2.527
PH2O = 0.467
