import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import GusSubroutines as gs
import Parameters as pr

plt.style.use('bmh')


class PhaseRetrievalSimulator:
    def __init__(self, theta_E=2, amp_E=0.1, phi_E2=0.0001):
        self.theta_E = theta_E
        self.amp_E = amp_E
        self.phi_E2 = phi_E2

        '''phi = gs.data_norm((phi2 - phi1))*60
        phi = gs.data_norm((phi2 - phi1))*150
        phi2 = gs.data_norm((phi2 - phi1))*150'''
        
        self.Ar = pr.Ar
        self.Ap = pr.Ap
        self.phi1 = pr.phi1
        self.phi2 = pr.phi2

        self.phi = gs.data_norm((self.phi2 - self.phi1))*60
        self.phi = gs.data_norm((self.phi2 - self.phi1))*150
        self.phi2 = gs.data_norm((self.phi2 - self.phi1))*150
        self.phi_w = np.arctan2(np.sin(self.phi), np.cos(self.phi))
        self.Error_function = pr.Error_function

        self.row = 100
        self.Ar = self.Ar[self.row, :]
        self.Ap = self.Ap[self.row, :]
        self.phi1 = self.phi1[self.row, :]
        self.phi2 = self.phi2[self.row, :]
        self.phi = self.phi[self.row, :]
        self.phi_w = self.phi_w[self.row, :]
        self.Error_function = gs.data_norm(self.Error_function[:, self.row])
        
        '''self.Ar = pr.Ar[100, :]
        self.Ap = pr.Ap[100, :]
        self.phi1 = pr.phi1[100, :]
        self.phi2 = pr.phi2[100, :]
        self.phi = gs.data_norm((self.phi2 - self.phi1)) * 60
        self.phi = gs.data_norm((self.phi2 - self.phi1)) * 150
        self.phi2 = gs.data_norm((self.phi2 - self.phi1)) * 150
        self.phi_w = np.arctan2(np.sin(self.phi), np.cos(self.phi))
        self.Error_function = gs.data_norm(pr.Error_function[:, 100])'''

        self.s1, self.s2, self.s3, self.s4 = 0, 90, 33, -33
        self.alpha0 = np.deg2rad(self.s1)
        self.alpha90 = np.deg2rad(self.s2)
        self.alpha1_ref = np.deg2rad(self.s3)
        self.alpha1_mean = 0

    def Qn(self, sigma):
        return [1, np.cos(2 * sigma), np.sin(2 * sigma)]

    def In(self, Ar, Ap, phi_r, phi_p, alpha, sigma, n):
        I = Ar ** 2 * (np.cos(alpha)) ** 2 + \
            (Ap * Ar * np.cos(alpha) *
             (-np.sin(alpha + phi_p - phi_r) + np.sin(alpha - 2 * sigma + phi_p - phi_r)
              + 2 * np.cos(sigma) * np.sin(alpha - sigma - phi_p + phi_r)) / np.sqrt(2)) + \
            1 / 4 * Ap ** 2 * (2 + np.cos(2 * alpha) - np.cos(2 * (alpha - 2 * sigma)))
        return I

    def generate_noise(self):
        Ar_noise = np.random.normal(self.Ar, self.Ar * self.amp_E)
        Ap_noise = np.random.normal(self.Ap, self.Ap * self.amp_E)
        noise = self.phi_E2 * np.asarray(random.sample(range(0, 1000), len(self.Error_function)))
        phi2_noise = self.Error_function * np.deg2rad(np.random.normal(0, self.phi_E2)) + noise
        return Ar_noise, Ap_noise, phi2_noise

    def alpha_from_I(self, Ii, Ii_1, I0, I1):
        alpha_ = (Ii + Ii_1 - I0 - I1) / (I0 - I1)
        alpha_ = np.nan_to_num(alpha_, nan=0.0, posinf=1e10, neginf=-1e10)
        alpha_ = np.clip(alpha_, -1.0, 1.0)
        alpha_ = np.nan_to_num(np.mean(alpha_))
        return 0.5 * np.arccos(alpha_)

    def run_simulation(self):
        N = 1000
        Ar_mean = 0
        Ap_mean = 0
        cos_mean = 0
        sin_mean = 0
        alpha1_mean = 0

        alpha1_list = []
        Ap_list = []
        Ar_list = []
        cos_list = []
        sin_list = []
        step_sum = np.zeros(N)

        theta_error = np.deg2rad(self.theta_E)

        for i in range(N):
            Ar0, Ap0, phi2_0 = self.generate_noise()
            I0 = self.In(Ar0, Ap0, self.phi1, self.phi2 + phi2_0, self.alpha0, theta_error, 0).real

            Ar1, Ap1, phi2_1 = self.generate_noise()
            I1 = self.In(Ar1, Ap1, self.phi1, self.phi2 + phi2_1, self.alpha90, theta_error, 0).real

            Ar2, Ap2, phi2_2 = self.generate_noise()
            I2 = self.In(Ar2, Ap2, self.phi1, self.phi2 + phi2_2, np.deg2rad(self.s3), theta_error, 0).real

            Ar3, Ap3, phi2_3 = self.generate_noise()
            I3 = self.In(Ar3, Ap3, self.phi1, self.phi2 + phi2_3, np.deg2rad(self.s4), theta_error, 0).real

            alpha1_ = self.alpha_from_I(I2, I3, I0, I1)
            step_sum[i] = alpha1_

            Q0 = self.Qn(self.alpha0)
            Q1 = self.Qn(self.alpha90)
            Q2 = self.Qn(alpha1_)
            Q3 = self.Qn(-alpha1_)

            Inm = np.array([I0, I1, I2, I3])
            Qnm = np.matrix([Q0, Q1, Q2, Q3])

            U = gs.least_squares(Inm, Qnm)
            u1, u2, u3 = U[0], U[1], U[2]

            alpha1_val = alpha1_ + np.zeros(u1.shape)
            Ap_val = np.nan_to_num(np.sqrt(2 * u1 - 2 * u2))
            Ar_val = np.nan_to_num(np.sqrt(2 * u1 - 2 * np.sqrt(u1 ** 2 - u2 ** 2 - u3 ** 2).real))

            phi_aux = np.nan_to_num(np.arctan2((u1 - u2 - np.sqrt(u1 ** 2 - u2 ** 2 - u3 ** 2).real), u3))
            cos_val = np.cos(phi_aux)
            sin_val = np.sin(phi_aux)

            alpha1_list.append(alpha1_val)
            Ap_list.append(Ap_val)
            Ar_list.append(Ar_val)
            cos_list.append(cos_val)
            sin_list.append(sin_val)

            alpha1_mean += alpha1_val
            Ap_mean += Ap_val
            Ar_mean += Ar_val
            cos_mean += cos_val
            sin_mean += sin_val

        alpha1_mean /= N
        Ap_mean /= N
        Ar_mean /= N
        cos_mean /= N
        sin_mean /= N
        phi_r = np.arctan2(sin_mean, cos_mean)

        phi_std = 0
        for k in range(N):
            cos_phij = cos_mean * cos_list[k] + sin_mean * sin_list[k]
            sin_phij = sin_mean * cos_list[k] - cos_mean * sin_list[k]
            phi_std += np.angle(cos_phij + 1j * sin_phij) ** 2
        phi_std = np.sqrt(1 / (N - 1) * phi_std)

        step_mean = np.mean(step_sum)
        step_std = np.std(step_sum)

        print(np.degrees(step_mean))

        t = np.linspace(0, len(Ar_mean), len(Ar_mean))
        fig = plt.figure()
        fig = plt.gcf()
        fig.suptitle("Phase step and complex amplitudes retrieved", fontsize=35)
        print('Mean', np.mean(phi_r))
        print('Teorethical', np.mean(self.phi_w))
        print('Diff', np.mean(phi_r)-np.mean(self.phi_w))
        # A_rA
        ax = plt.subplot(221)
        ax.set_title(r"(a)                          $\alpha_{rA}$", fontsize=30)
        ax.set_xticks([])
        ax.set_yticklabels(['$0$', '$\dfrac{\pi}{2}$'])
        plt.yticks(fontsize='20')
        ax.set_xticks([0, 200, 400, 600, 800, 1000])
        plt.errorbar(t, (self.alpha1_ref) + np.zeros_like(t), color='black', linestyle='dotted', linewidth=3)
        plt.errorbar(t, alpha1_mean, color='#c23728', linewidth=3, linestyle='dashed', alpha=0.8)
        plt.errorbar(t, alpha1_mean, yerr=(self.alpha1_ref) * step_std, linewidth=0.3, ecolor='#1984c5')
        plt.legend([r"$\alpha$", r"Mean", "Standard deviation"], loc="best", prop={'size': 15})
        plt.yticks([0, np.pi/2], fontsize='20')

        # A_rA
        ax = plt.subplot(222)
        ax.set_title(r"(b)                          $A_{rA}$", fontsize=30)
        ax.set_xticks([])
        plt.errorbar(t, self.Ar, color='black', linestyle='dotted', linewidth=3)
        plt.errorbar(t, Ar_mean, color='#c23728', linewidth=3, linestyle='dashed', alpha=0.8)
        plt.errorbar(t, Ar_mean, yerr=np.std(Ar_list, axis=0), linewidth=0.2, ecolor='#1984c5')
        plt.legend([r"$A_{r}$", "Mean", "Standard deviation"], loc="best", prop={'size': 15})
        plt.yticks([0.1, 1.3], fontsize='13')

        # A_pA
        ax = plt.subplot(223)
        ax.set_title(r"(d)                          $A_{pA}$", fontsize=30)
        plt.errorbar(t, self.Ap, color='black', linestyle='dotted', linewidth=3)
        plt.errorbar(t, Ap_mean, color='#c23728', linewidth=3, linestyle='dashed', alpha=0.8)
        plt.errorbar(t, Ap_mean, yerr=np.std(Ap_list, axis=0), linewidth=0.2, ecolor='#1984c5')
        plt.legend([r"$A_{p}$", "Mean", "Standard deviation"], loc="best", prop={'size': 15})
        plt.yticks([0.9, 2.0], fontsize='13')

        # phi_wA
        ax = plt.subplot(224)
        ax.set_title(r"(d)                          $\phi_{wA}$", fontsize=30)
        plt.errorbar(t, self.phi_w, color='black', linestyle='dotted', linewidth=3)
        plt.errorbar(t, phi_r, color='#c23728', linewidth=3, linestyle='dashed', alpha=0.8)
        plt.errorbar(t, phi_r, yerr=phi_std, linewidth=0.2, ecolor='#1984c5')
        plt.legend([r"$\phi_{w}$", "Mean", "Standard deviation"], loc="best", prop={'size': 15})
        plt.yticks([-np.pi, np.pi], fontsize='20')
        ax.set_yticklabels(['$-\pi$', '$\pi$'])

        gs.show()


if __name__ == "__main__":
    simulator = PhaseRetrievalSimulator(theta_E=2, amp_E=0.03, phi_E2=0.00001)
    simulator.run_simulation()
