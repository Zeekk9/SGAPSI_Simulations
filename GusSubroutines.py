import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import dct, idct
import sys
#several useful functions for analysis

def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def solve_poisson(rho):
    N, M = rho.shape
    dct_rho = dct(dct(rho.T, norm='ortho').T, norm='ortho')
    I, J = np.meshgrid(np.arange(M), np.arange(N))
    denom = 2 * (np.cos(np.pi * I / M) + np.cos(np.pi * J / N) - 2)
    denom[0, 0] = 1  # Avoid division by zero
    dct_phi = dct_rho / denom
    dct_phi[0, 0] = 0  # Set the mean to zero
    phi = idct(idct(dct_phi.T, norm='ortho').T, norm='ortho')
    return phi

def apply_q(p, ww):
    dx = np.concatenate([np.diff(p, axis=1), np.zeros((p.shape[0], 1))], axis=1)
    dy = np.concatenate([np.diff(p, axis=0), np.zeros((1, p.shape[1]))], axis=0)

    ww_dx = ww * dx
    ww_dy = ww * dy

    ww_dx2 = np.concatenate([np.zeros((p.shape[0], 1)), ww_dx], axis=1)
    ww_dy2 = np.concatenate([np.zeros((1, p.shape[1])), ww_dy], axis=0)

    q_p = np.diff(ww_dx2, axis=1) + np.diff(ww_dy2, axis=0)
    return q_p

def phase_unwrap(psi, weight=None):
    if weight is None:
        dx = np.concatenate([np.zeros((psi.shape[0], 1)), wrap_to_pi(np.diff(psi, axis=1)), np.zeros((psi.shape[0], 1))], axis=1)
        dy = np.concatenate([np.zeros((1, psi.shape[1])), wrap_to_pi(np.diff(psi, axis=0)), np.zeros((1, psi.shape[1]))], axis=0)
        rho = np.diff(dx, axis=1) + np.diff(dy, axis=0)
        phi = solve_poisson(rho)
    else:
        if psi.shape != weight.shape:
            raise ValueError("Weight must be the same shape as the input phase")
        
        dx = np.concatenate([wrap_to_pi(np.diff(psi, axis=1)), np.zeros((psi.shape[0], 1))], axis=1)
        dy = np.concatenate([wrap_to_pi(np.diff(psi, axis=0)), np.zeros((1, psi.shape[1]))], axis=0)

        ww = weight ** 2
        ww_dx = ww * dx
        ww_dy = ww * dy

        ww_dx2 = np.concatenate([np.zeros((psi.shape[0], 1)), ww_dx], axis=1)
        ww_dy2 = np.concatenate([np.zeros((1, psi.shape[1])), ww_dy], axis=0)

        rk = np.diff(ww_dx2, axis=1) + np.diff(ww_dy2, axis=0)
        norm_r0 = np.linalg.norm(rk)
        phi = np.zeros_like(psi)
        eps = 1e-8
        k = 0

        while np.any(rk != 0):
            zk = solve_poisson(rk)
            if k == 0:
                pk = zk
            else:
                beta_k = np.sum(rk * zk) / np.sum(rk_prev * zk_prev)
                pk = zk + beta_k * pk

            rk_prev = rk
            zk_prev = zk

            q_pk = apply_q(pk, ww)
            alpha_k = np.sum(rk * zk) / np.sum(pk * q_pk)

            phi += alpha_k * pk
            rk -= alpha_k * q_pk

            if k >= psi.size or np.linalg.norm(rk) < eps * norm_r0:
                break
            k += 1
    return phi


def Seidel(x, y, a, b, c, d, e, f, g):
    s = a*((x**2)+(y**2))**2+b*((x**2)+(y**2))*(y)+c * \
        ((x**2)+(3*(y**2)))+d * \
        ((x**2)+(y**2))+(e*(y))+(f*(x))+g
    return s


def csc(sigma):
    csc = 1/np.sin(sigma)
    return csc


def sec(sigma):
    sec = 1/np.cos(sigma)
    return sec


def cot(sigma):
    cot = np.cos(sigma)/np.sin(sigma)
    return cot


def c_matx(n, m, error, shape):
    c = np.zeros(shape)+1/4*np.sinc(1/2*n)*np.sinc(1/2*m)
    c = np.random.normal(c, c*error, np.shape(shape))
    return c


def c1_matx(n, m, error, shape):
    c = np.zeros(shape)+1/4*np.sinc(1/2*(n+1))*np.sinc(1/2*m)
    c = np.random.normal(c, c*error, shape)
    return c

def least_squares(Inm, Qnm):
    Qnm = np.asarray(Qnm)
    Inm = np.asarray(Inm)
    A = np.linalg.inv(Qnm.T @ Qnm)
    B = Qnm.T @ Inm
    U = A @ B
    return U


def c(n, m):
    c = 0.5*np.sinc(0.5*n)*0.5*np.sinc(0.5*m)
    return c


def gradtorad(x):
    r = x*np.pi/180
    return r

def radtograd(r):
    x=r*180/np.pi
    return x


def itoh_2D(W):
    renglon, columna = W.shape
    phi = np.zeros(W.shape)
    psi = np.zeros(W.shape)
    phi[0, 0] = W[0, 0]
    # Se Desenvuelve la primera columna
    for m in range(1, columna):
        Delta = W[0, m] - W[0, m - 1]
        WDelta = np.arctan2(np.sin(Delta), np.cos(Delta))
        phi[0, m] = phi[0, m - 1] + WDelta
    psi[0, :] = phi[0, :]

    for k in range(columna):
        psi[0, k] = W[0, k]
        for p in range(1, renglon):
            Delta = W[p, k] - W[p - 1, k]
            WDelta = np.arctan2(np.sin(Delta), np.cos(Delta))
            phi[p, k] = phi[p - 1, k] + WDelta
    return phi


def matshow(position, mat, title):
    """"Subroutine to show matrix"""
    """1: postion, 2: matrix, 3: Title"""
    plt.subplot(position)
    plt.imshow(mat, cmap='gist_heat')
    plt.title(title, fontsize=30)
    plt.axis('off')
    '''cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)'''
    plt.colorbar()


def surf(p1, p2, p3, W, title,fig):
    xlim = W[0, :].size
    ylim = W[:, 0].size
    x = np.linspace(0, xlim, xlim)
    y = np.linspace(0, ylim, ylim)
    X, Y = np.meshgrid(x, y)
    #fig.suptitle(title, fontsize=40)
    ax = fig.add_subplot(p1, p2, p3, projection='3d')
    plt.title(title, fontsize=30)
    ax.plot_surface(X, Y, W, cmap='gist_heat')
    plt.axis('on')


def wrap(W):
    W = np.arctan2(np.sin(W), np.cos(W))
    return W


def prom_filter(M):
    x, y = M.shape
    M_filter = np.zeros((x-1, y-1))
    Sum = 0
    # print(M.shape)
    for i in range(x-2):
        # print('cambio2')
        for j in range(y-2):
            # print('cambio')
            Sum = 0
            for k in range(3):
                for l in range(3):
                    #print('posicion:', i+k, j+k, ' Valor:', M[i+k, j+l])
                    Sum += M[i+k, j+l]
                    if k == 2 and l == 2:
                        M_filter[i+1, j+1] = Sum/9
                        # print(M[i+1,j+1])
                        # print(i+1,j+1)
    return M_filter[1:x-1, 1:y-1]


def mouse_crop(event, x, y, flags, param):
    global x_c, y_c
    if event == cv2.EVENT_LBUTTONDOWN:  # checks mouse left button down condition
        print("Coordinates of pixel: X: ", x, "Y: ", y)
        x_c, y_c = x, y


def crop(image, ancho, largo, n):
    Is = []
    cord = []
    alpha = 1  # Contrast control
    beta = 1  # Brightness control
    croped = image
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    for i in range(0, n):

        #x_c, y_c = 0, 0
        cropping = False
        #image = inter
        #oriImage = image.copy()

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("image", mouse_crop)
        #cv2.resizeWindow('image', 1280, 800)

        while True:
            i = image.copy()
            key = cv2.waitKey(2)
            if not cropping:
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.imshow("image", image)

            if key % 256 == 27:
                cv2.destroyAllWindows()
                break

        # Is.append(np.mean(cv2.fastNlMeansDenoising(image[y_c-2*ancho:y_c,
        #                        x_c-largo:x_c+largo]), axis=2)*1.0)
        Is.append(np.mean((croped[y_c-2*ancho:y_c,
                                  x_c-largo:x_c+largo]), axis=2)*1.0)
        cord.append([y_c, x_c])
        image[y_c-2*ancho:y_c, x_c-largo:x_c+largo] = 0

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()

    return Is, cord


def ROI(image_name):
    
    def mouse_crop(event, x, y, flags, param):
        # grab references to the global variables
        global x_start, y_start, x_end, y_end, cropping

        # if the left mouse button was DOWN, start RECORDING
        # (x, y) coordinates and indicate that cropping is being
        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True

        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping == True:
                x_end, y_end = x, y

        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates
            x_end, y_end = x, y
            cropping = False  # cropping is finished

            refPoint = [(x_start, y_start), (x_end, y_end)]
            print('Coordenadas')
            print('x_start=', x_start)
            print('y_start=', y_start)
            print('x_end=', x_end)
            print('y_end=', y_end)
            if len(refPoint) == 2:  # when two points were found
                roi = oriImage[refPoint[0][1]:refPoint[1]
                            [1], refPoint[0][0]:refPoint[1][0]]
                cv2.imshow("Cropped", roi)


    cropping = False
    #x_start, y_start, x_end, y_end = 0, 0, 0, 0
    image = image_name
    oriImage = image.copy()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", mouse_crop)
    cv2.resizeWindow('image', 1280, 800)

    while True:

        i = image.copy()
        key = cv2.waitKey(2)
        if not cropping:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow("image", image)

        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow("image", i)

        if key % 256 == 27:
            cv2.destroyAllWindows()
            break
        
    return x_start, y_start, x_end, y_end
            

def show():
    """Maximiza la ventana de matplotlib según el sistema operativo."""
    if sys.platform.startswith('win'):
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
    elif sys.platform.startswith('linux'):
        manager.window.showMaximized()  # Linux (puede requerir TkAgg)
    elif sys.platform.startswith('darwin'):  # macOS
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()  # Funciona en varios backends, incluido macOS
    else:
        print("Sistema operativo no soportado para maximización.")

    plt.show()
    
def smooth(Original, Retrieved):
    alpha = 0.55  # peso de imagen1
    beta = 1 - alpha  # peso de imagen2
    blended = alpha * Original + beta * Retrieved
    return blended

def data_norm(data):
    data_norm = (data - data.min()) / (data.max() - data.min())
    return data_norm

def smooth(Original, Original_Weight, Retrieved):
    #Original_Weight = 0.55  # peso de imagen1
    Retrieved_Weight = 1 - Original_Weight  # peso de imagen2
    blended = Original_Weight * Original + Retrieved_Weight * Retrieved
    return blended
