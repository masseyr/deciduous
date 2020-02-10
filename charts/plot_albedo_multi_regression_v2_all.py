import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sphviewer as sph


def myplot(x, y, nb=32, xsize=500, ysize=500):
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    x0 = (xmin + xmax) / 2.
    y0 = (ymin + ymax) / 2.

    pos = np.zeros([len(x), 3])
    pos[:, 0] = x
    pos[:, 1] = y
    w = np.ones(len(x))

    P = sph.Particles(pos, w, nb=nb)
    S = sph.Scene(P)
    S.update_camera(r='infinity', x=x0, y=y0, z=0,
                    xsize=xsize, ysize=ysize)
    R = sph.Render(S)
    R.set_logscale()
    img = R.get_image()
    extent = R.get_extent()
    for i, j in zip(xrange(4), [x0, x0, y0, y0]):
        extent[i] += j

    return img, extent


if __name__ == '__main__':
    fig = plt.figure(1, figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # Generate some test data
    x = np.random.randn(1000)
    y = np.random.randn(1000)

    # Plotting a regular scatter plot
    ax1.plot(x, y, 'k.', markersize=5)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)

    heatmap_8, extent_8 = myplot(x, y, nb=6)
    heatmap_16, extent_16 = myplot(x, y, nb=8)
    heatmap_32, extent_32 = myplot(x, y, nb=12)

    ax2.imshow(heatmap_8, extent=extent_8, origin='lower', aspect='auto', cmap=cm.hot_r)
    ax2.set_title("Smoothing over 6 neighbors")

    ax3.imshow(heatmap_16, extent=extent_16, origin='lower', aspect='auto', cmap=cm.hot_r)
    ax3.set_title("Smoothing over 8 neighbors")

    ax4.imshow(heatmap_32, extent=extent_32, origin='lower', aspect='auto', cmap=cm.hot_r)
    ax4.set_title("Smoothing over 12 neighbors")

    plt.savefig('example3.png')
