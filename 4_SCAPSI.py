import numpy as np
import GusSubroutines as gs
import Parameters as pr
import random
import matplotlib.pyplot as plt


class SCPASIInterferogram:
    def __init__(self, Ar, Ap, phi, A_noise_level=0.01, phi_noise_level=1e-7, row=100):
        self.Ar = Ar
        self.Ap = Ap
        self.phi = phi
        self.A_noise_level = A_noise_level
        self.phi_noise_level = phi_noise_level
        self.row = row

    def gradtorad(self, x):
        return x * np.pi / 180

    def add_noise(self, data, noise_level):
        return np.random.normal(data, data * noise_level)

    def add_phi_noise(self):
        x = np.linspace(0.5, 3, self.phi.shape[1])
        y = np.linspace(0.5, 3, self.phi.shape[0])
        X, Y = np.meshgrid(x, y)
        error_function = X**3 + Y**3 + 2 * X**4 * np.cos(X**3) * 4 * np.exp(X**3 * Y**2)
        noise = (self.phi_noise_level * 0.00004) * np.random.rand(*self.phi.shape)
        phi_noise = error_function * gs.gradtorad(np.random.normal(0, self.phi_noise_level)) + noise
        return self.phi + phi_noise

    def interferogram(self, Ar, Ap, phi, alpha):
        return Ar**2 * np.cos(alpha)**2 + 0.5 * Ap**2 + np.sqrt(2) * Ar * Ap * np.cos(alpha) * np.cos(phi + alpha)

    def compute_alpha(self, I_i, I_i1, I0, I1):
        alpha_ = (I_i + I_i1 - I0 - I1) / (I0 - I1 + 1e-10)
        alpha_ = np.nan_to_num(alpha_, nan=0.0, posinf=1e10, neginf=-1e10)
        alpha_ = np.clip(alpha_, -1.0, 1.0)
        return 0.5 * np.arccos(np.nan_to_num(np.mean(alpha_)))

    def get_Qn(self, alpha):
        return [1, np.cos(2*alpha), np.sin(2*alpha)]

    def recover_phase(self):
        alpha_fixed = self.gradtorad(16)

        I = []
        for shift in [0, self.gradtorad(90), alpha_fixed, -alpha_fixed]:
            Ar_n = self.add_noise(self.Ar, self.A_noise_level)
            Ap_n = self.add_noise(self.Ap, self.A_noise_level)
            phi_n = self.add_phi_noise()
            I.append(self.interferogram(Ar_n, Ap_n, phi_n, shift))

        I0, I1, I2, I3 = I
        alpha1 = self.compute_alpha(I2, I3, I0, I1)

        Q = np.matrix([
            self.get_Qn(0),
            self.get_Qn(self.gradtorad(90)),
            self.get_Qn(alpha_fixed),
            self.get_Qn(-alpha_fixed)
        ])

        In_stack = np.array([I0, I1, I2, I3])
        U = gs.least_squares(In_stack, Q)

        u1, u2, u3 = U
        phi_rec = np.nan_to_num(-np.arctan2(np.sin(np.arctan2((u1 - u2 - np.sqrt(u1**2 - u2**2 - u3**2)), u3) + np.pi/2),
                              np.cos(np.arctan2((u1 - u2 - np.sqrt(u1**2 - u2**2 - u3**2)), u3) + np.pi/2)))
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

# Arreglos para guardar resultados
phi_SCAPSI_all = np.zeros((N_RUNS, cols))

# Ejecución
for i in range(N_RUNS):
    SCAPSI = SCPASIInterferogram(Ar, Ap, phi, A_noise_level, phi_noise_level, row)
    phi_rec = SCAPSI.recover_phase()
    phi_SCAPSI_all[i, :] = phi_rec[row, :]
    print(f"Simulación {i+1}/{N_RUNS} completada")
        

# Guardar en archivo
np.savez('SCAPSI_results_row100.npz', phi_true=phi_true, phi_SCAPSI_all=phi_SCAPSI_all)
print("Resultados guardados en 'SCAPSI_results_row100.npz'")