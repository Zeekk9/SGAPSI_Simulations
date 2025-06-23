import numpy as np
import matplotlib.pyplot as plt
import Parameters as pr
import GusSubroutines as gs

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
        # Generar interferogramas con ruido para las 4 fases
        I1 = self.interferogram(self.add_noise_to_Ar(), self.add_noise_to_Ap(), self.add_noise_to_phi(), -3 * self.sigma)
        I2 = self.interferogram(self.add_noise_to_Ar(), self.add_noise_to_Ap(), self.add_noise_to_phi(), -self.sigma)
        I3 = self.interferogram(self.add_noise_to_Ar(), self.add_noise_to_Ap(), self.add_noise_to_phi(), self.sigma)
        I4 = self.interferogram(self.add_noise_to_Ar(), self.add_noise_to_Ap(), self.add_noise_to_phi(), 3 * self.sigma)

        # Demodulación Carré
        numerator = 3 * (I2 - I3) - (I1 - I4)
        denominator = I1 + I2 - I3 - I4 + 1e-10  # Previene división por cero

        delta_est = np.arctan(np.sqrt(numerator / denominator))

        num = (I1 - I4 + I2 - I3) * np.tan(delta_est)
        den = I2 + I3 - I1 - I4

        phi_rec = np.nan_to_num(np.arctan2(num, den))
        return phi_rec
    

'''Ar = pr.Ar
Ap = pr.Ap
phi = pr.phi
A_noise_level=0.03
phi_noise_level=1e-6
row=100

carre_runner = CarreInterferogram(Ar, Ap, phi, A_noise_level, phi_noise_level, row)
phi_rec = carre_runner.recover_phase()

# Visualización
row = 100
plt.plot(gs.wrap(phi)[row, :], label='Fase teórica')
plt.plot(phi_rec[row, :], label='Carré')
plt.legend()
gs.show()
'''

# Configuración
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

# Arreglos para guardar resultados
phi_carre_all = np.zeros((N_RUNS, cols))

# Ejecución
for i in range(N_RUNS):
    carre = CarreInterferogram(Ar, Ap, phi, A_noise_level, phi_noise_level, row)
    phi_rec = carre.recover_phase()
    phi_carre_all[i, :] = phi_rec[row, :]
    print(f"Simulación {i+1}/{N_RUNS} completada")
        

# Guardar en archivo
np.savez('carre_results_row100.npz', phi_true=phi_true, phi_carre_all=phi_carre_all)
print("Resultados guardados en 'carre_results_row100.npz'")