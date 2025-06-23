# SGAPSI_Simulations

**Numerical Simulations, Noise Analysis, and Comparisons of the SG-APSI Method**

This repository contains code used for the numerical analysis, noise robustness evaluation, and comparison of the **N-Step Self-Calibrated Generalized Amplitude-Phase-Shifting Interferometry (SG-APSI)** technique.

---

## 📦 Requirements

- Python 3.11.7  
- Recommended: [Anaconda](https://www.anaconda.com/) distribution  
- Alternatively, install the following packages manually:
  - `numpy`
  - `matplotlib`
  - `random`
  - `time`

---

## 🗂 File Structure & Usage

- **`1_Carre.py`**, **`2_PCA.py`**, **`3_VES.py`**, **`4_SCAPSI.py`**  
  Implementations of phase-retrieval methods:
  - Carré Method
  - Principal Component Analysis (PCA)
  - Volume Enclosed by a Surface (VES)
  - Self-Calibrated SG-APSI (proposed method)

  Each script generates simulation data stored in `.npz` files.

- **`0_Comparation_Plots.py`**  
  Loads and compares the results from the four methods using the generated datasets.  
  > ⚠️ If you change the noise levels, **re-run all four method scripts** before executing this script.

---

## 🔧 Configuring Noise Levels

Noise parameters can be customized directly in each method script:

```python
A_noise_level = 0.01        # Amplitude noise (e.g., 0.01 = 10% error)
phi_noise_level = 1e-9      # Phase noise (e.g., ~10 degrees)
row = 100                   # Row index used for simulation and analysis


# ----------------------------------------------------------------------
# SGAPSI Simulations: Intelligent Optical Field Retrieval with PSI
# Copyright (c) 2025 Carlos Augusto Flores Meneses
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# ----------------------------------------------------------------------
