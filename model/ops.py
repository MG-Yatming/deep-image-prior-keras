import numpy as np

def lanczos2_kernel(factor, phase=0.5):
    '''
    The parameter a is a positive integer, typically 2 or 3, which determines the size of the kernel.
    The Lanczos kernel has 2a - 1 lobes: a positive one at the center,
    and a - 1 alternating negative and positive lobes on each side.
    '''
    a = 2
    kernel_size = 4 * factor + 1

    if phase == 0.5:
        kernel = np.zeros([kernel_size - 1, kernel_size - 1])

    center = (kernel_size + 1) / 2.

    for i in range(1, kernel.shape[0] + 1):
        for j in range(1, kernel.shape[1] + 1):
            if phase == 0.5:
                di = abs(i + 0.5 - center) / factor
                dj = abs(j + 0.5 - center) / factor
            else:
                di = abs(i - center) / factor
                dj = abs(j - center) / factor

            val = 1
            if di != 0:
                val = val * a * np.sin(np.pi * di) * np.sin(np.pi * di / a)
                val = val / (np.pi * np.pi * di * di)

            if dj != 0:
                val = val * a * np.sin(np.pi * dj) * np.sin(np.pi * dj / a)
                val = val / (np.pi * np.pi * dj * dj)

            kernel[i - 1][j - 1] = val

    kernel /= kernel.sum()

    return kernel