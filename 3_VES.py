import numpy as np
import matplotlib.pyplot as plt
import Parameters as pr
import GusSubroutines as gs
'''
VES Algorithm:

@article{Meneses-Fabian_2016,
doi = {10.1088/2040-8978/18/12/125703},
url = {https://dx.doi.org/10.1088/2040-8978/18/12/125703},
year = {2016},
month = {nov},
publisher = {IOP Publishing},
volume = {18},
number = {12},
pages = {125703},
author = {Meneses-Fabian, Cruz},
title = {Self-calibrating generalized phase-shifting interferometry of three phase-steps based on geometric concept of volume enclosed by a surface},
journal = {Journal of Optics},
abstract = {This paper presents a non-iterative, fast, and simple algorithm for phase retrieval, in phase-shifting interferometry of three unknown and unequal phase-steps, based on the geometric concept of the volume enclosed by a surface. This approach can be divided in three stages; first the background is eliminated by the subtraction of two interferograms, for obtaining a secondary pattern; second, a surface is built by the product of two secondary patterns and the volume enclosed by this surface is computed; and third, the ratio between two enclosed volumes is approximated to a constant that depends on the phase-steps, with which a system of equations is established, and its solution allows the measurement of the phase-steps to be obtained. Additional advantages of this approach are its immunity to noise, and its capacity to support high spatial variations in the illumination. This approach is theoretically described and is numerically and experimentally verified.}
}
'''
class VESInterferogram:
    def __init__(self, Ar, Ap, phi, A_noise_level=0.0, phi_noise_level=0.0, row=100):
        self.Ar = Ar
        self.Ap = Ap
        self.phi = phi
        self.A_noise_level = A_noise_level
        self.phi_noise_level = phi_noise_level
        self.row = row
        self.sigma1 = np.deg2rad(60)
        self.sigma2 = np.deg2rad(160)

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

    def interferogram(self, Ar, Ap, phi, sigma):
        return Ar**2 + Ap**2 + 2 * Ar * Ap * np.cos(phi + sigma)

    def recover_phase(self):
        # Interferogramas con ruido
        Ar0 = self.add_noise_to_Ar()
        Ap0 = self.add_noise_to_Ap()
        phi0 = self.add_noise_to_phi()
        I0 = self.interferogram(Ar0, Ap0, phi0, 0)

        Ar1 = self.add_noise_to_Ar()
        Ap1 = self.add_noise_to_Ap()
        phi1 = self.add_noise_to_phi()
        I1 = self.interferogram(Ar1, Ap1, phi1, self.sigma1)

        Ar2 = self.add_noise_to_Ar()
        Ap2 = self.add_noise_to_Ap()
        phi2 = self.add_noise_to_phi()
        I2 = self.interferogram(Ar2, Ap2, phi2, self.sigma2)

        # VES demodulación
        p = I0 - I1
        q = I1 - I2
        r = I0 - I2

        A1 = np.trace(q.T @ r)
        B1 = np.trace(q.T @ q) * np.trace(r.T @ r)

        A2 = np.trace(p.T @ r)
        B2 = np.trace(p.T @ p) * np.trace(r.T @ r)

        alpha1 = 2 * np.arccos(A1 / np.sqrt(B1))
        alpha2 = 2 * np.nan_to_num(np.arccos(A2 / np.sqrt(B2))) + alpha1

        phi_rec = np.nan_to_num(np.arctan2(
            I2 - I1 + (I0 - I2) * np.cos(alpha1) - (I0 - I1) * np.cos(alpha2),
            (I0 - I2) * np.sin(alpha1) - (I0 - I1) * np.sin(alpha2)
        ))

        return phi_rec
    
N_RUNS = 1000
row = 100
phi_true = gs.wrap(pr.phi)[row, :]
cols = phi_true.shape[0]
Ar = pr.Ar
Ap = pr.Ap
phi = pr.phi
A_noise_level=10e-2 #10% of error
phi_noise_level=10e-9#10 degrees of error
row=100

ves_runner = VESInterferogram(Ar, Ap, phi, A_noise_level, phi_noise_level, row=100)
phi_rec = ves_runner.recover_phase()

'''plt.plot(gs.wrap(phi)[row, :], label='Fase teórica')
plt.plot(phi_rec[row, :], label='VES')
plt.legend()
gs.show()'''


# Arreglos para guardar resultados
phi_ves_all = np.zeros((N_RUNS, cols))

# Ejecución
for i in range(N_RUNS):
    ves = VESInterferogram(Ar, Ap, phi, A_noise_level, phi_noise_level, row)
    phi_rec = ves.recover_phase()
    phi_ves_all[i, :] = phi_rec[row, :]
    print(f"Simulación {i+1}/{N_RUNS} completada")
        

# Guardar en archivo
np.savez('VES_results_row100.npz', phi_true=phi_true, phi_ves_all=phi_ves_all)
print("Resultados guardados en 'ves_results_row100.npz'")
