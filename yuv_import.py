import numpy as np


def yuv_import(filename, dims, numfrm, startfrm):
    fp = open(filename, 'rb')
    blk_size = np.prod(dims) * 3 / 2
    fp.seek(np.int(blk_size * startfrm), 0)

    # print dims[0]
    # print dims[1]
    d00 = dims[0] // 2
    d01 = dims[1] // 2

    Y = np.zeros((numfrm, dims[0], dims[1]), np.uint8, 'C')
    U = np.zeros((numfrm, d00, d01), np.uint8, 'C')
    V = np.zeros((numfrm, d00, d01), np.uint8, 'C')

    # Yt = np.zeros((dims[0], dims[1]), np.uint8, 'C')
    # Ut = np.zeros((d00, d01), np.uint8, 'C')
    # Vt = np.zeros((d00, d01), np.uint8, 'C')
    for i in range(numfrm):
        for m in range(dims[0]):
            for n in range(dims[1]):
                # print m,n
                Y[i, m, n] = ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                U[i, m, n] = ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                V[i, m, n] = ord(fp.read(1))
        # Y = Y + [Yt]
        # U = U + [Ut]
        # V = V + [Vt]
        # Y[i, 0:dims[0], 0:dims[1]] = Yt[0:dims[0], 0:dims[1]]
        # U[i, 0:d00, 0:d01] = Ut[0:d00, 0:d01]
        # Y[i, 0:d00, 0:d01] = Vt[0:d00, 0:d01]
    fp.close()
    return (Y, U, V)

# def main(argv=None):  # pylint: disable=unused-argument
#   Y,U,V = yuv_import("/media/root/Lab_Sxl1/YUV_LDP_QP42/football_cif.yuv", (288,352),100,0)
#   y1 = Y[0]
#   y2 = Y[10]
#
#   plt.imshow(y1,cmap ='gray' )
#   plt.show()
#   plt.imshow(y2,cmap ='gray')
#   plt.show()
#
#
#
# if __name__ == '__main__':
#   tf.app.run()
