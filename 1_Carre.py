import numpy as np
import matplotlib.pyplot as plt
import Parameters as pr
import GusSubroutines as gs
'''
Carré algorithm:

@article{PCarre_1966,
doi = {10.1088/0026-1394/2/1/005},
url = {https://dx.doi.org/10.1088/0026-1394/2/1/005},
year = {1966},
month = {jan},
publisher = {},
volume = {2},
number = {1},
pages = {13},
author = {P Carré},
title = {Installation et utilisation du comparateur photoélectrique et interférentiel du Bureau International des Poids et Mesures},
journal = {Metrologia},
abstract = {The paper describes the photoelectric and interference comparator of the BIPM, for divided scales and end standards, together with its photoelectric microscopes, its installation (in particular, its anti-vibration mounting) and its auxiliary apparatus. The latter includes the interferometer, the refractometer and apparatus for temperature measurement. The results of the tests to which the complete assembly has been subjected are given, together with the results of the first measurements made with this new comparator.}
}
'''
class CarreInterferogram:
    def __init__(self, Ar, Ap, phi, A_noise_level=0.0, phi_noise_level=0.0, row=100):
        self.Ar = Ar
        self.Ap = Ap
        self.phi = phi
        self.A_noise_level = A_noise_level
        self.phi_noise_level = phi_noise_level
        self.row = row
        self.sigma = np.deg2rad(55)

    def add_noise_to_Ar(self):
        return np.random.normal(self.Ar, self.Ar * self.A_noise_level)

    def add_noise_to_Ap(self):
        return np.random.normal(self.Ap, self.Ap * self.A_noise_level)

    def add_noise_to_phi(self):
        x = np.linspace(0.5, 3, self.phi.shape[1])
        y = np.linspace(0.5, 3, self.phi.shape[0])
        X, Y = np.meshgrid(x, y)
        error_function = X**3 + Y**3 + 2 * X**4 * np.cos(X**3) * 4 * np.exp(X**3 * Y**2)
        noise = (self.phi_noise_level * 0.00004) * np.random.rand(*self.phi.shape)
        phi_noise = error_function * gs.gradtorad(np.random.normal(0, self.phi_noise_level)) + noise
        return self.phi + phi_noise

    def interferogram(self, Ar, Ap, phi, alpha):
        return Ap**2 + Ar**2 + 2 * Ar * Ap * np.cos(phi + alpha)

    def recover_phase(self):
        
        I1 = self.interferogram(self.add_noise_to_Ar(), self.add_noise_to_Ap(), self.add_noise_to_phi(), -3 * self.sigma)
        I2 = self.interferogram(self.add_noise_to_Ar(), self.add_noise_to_Ap(), self.add_noise_to_phi(), -self.sigma)
        I3 = self.interferogram(self.add_noise_to_Ar(), self.add_noise_to_Ap(), self.add_noise_to_phi(), self.sigma)
        I4 = self.interferogram(self.add_noise_to_Ar(), self.add_noise_to_Ap(), self.add_noise_to_phi(), 3 * self.sigma)

        # Demodulación Carré
        numerator = 3 * (I2 - I3) - (I1 - I4)
        denominator = I1 + I2 - I3 - I4 + 1e-10  

        delta_est = np.arctan(np.sqrt(numerator / denominator))

        num = (I1 - I4 + I2 - I3) * np.tan(delta_est)
        den = I2 + I3 - I1 - I4

        phi_rec = np.nan_to_num(np.arctan2(num, den))
        return phi_rec
    


# Configuration parameters
N_RUNS = 1000
row = 100
phi_true = gs.wrap(pr.phi)[row, :]
cols = phi_true.shape[0]
Ar = pr.Ar
Ap = pr.Ap
phi = pr.phi
A_noise_level=0.01 #10% of error
phi_noise_level=1e-9 #10 degrees of error
row=100

# Results arrays
phi_carre_all = np.zeros((N_RUNS, cols))

# Implementation
for i in range(N_RUNS):
    carre = CarreInterferogram(Ar, Ap, phi, A_noise_level, phi_noise_level, row)
    phi_rec = carre.recover_phase()
    phi_carre_all[i, :] = phi_rec[row, :]
    print(f"Simulación {i+1}/{N_RUNS} completada")
        

# Saving data
np.savez('carre_results_row100.npz', phi_true=phi_true, phi_carre_all=phi_carre_all)
print("Resultados guardados en 'carre_results_row100.npz'")
