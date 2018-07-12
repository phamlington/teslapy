"""
"""
from math import *
import numpy as np
# from scipy.stats import norm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import importlib
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# mpl.use('PDF')
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True
mpl.rcParams['backend']='MacOSX'
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica Neue']})
# mpl.rc('font', **{'family': 'serif', 'serif': ['Times']})
mpl.rc('axes.formatter', use_mathtext=False, )
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['cm']
mpl.rc('axes', labelpad=3.0, )

plt = importlib.reload(plt)

np.set_printoptions(formatter={'float': '{: .8e}'.format})

C = ['#53011b', '#660117', '#790113', '#8b000f', '#9e000b', '#b10007',
     '#c30003', '#d30401', '#da1203', '#e12006', '#e82f09', '#ef3d0c',
     '#f54c0e', '#fc5a11', '#ff6d24', '#ff8547', '#ff9c6a', '#ffb48e',
     '#ffcbb1', '#ffe2d4', '#fffaf7', '#ffffff', '#e3e3ff', '#bdbdff',
     '#9898ff', '#7272ff', '#4c4cff', '#2626ff', '#0000ff']
cm = mpl.colors.ListedColormap(C)
mpl.cm.register_cmap(name='Skk_RdBu', cmap=cm)

# multiCase Colors
cc = ['#2166ac', '#4393c3', '#92c5de', '#dde094',
      '#ffdd7a', '#f4a582', '#d6604d', '#b2182b']
# cc = ['#4575b4', '#74add1', '#abd9e9', '#e0f3f8',
#       '#fee090', '#fdae61', '#f46d43', '#d73027']
# cc = ['#4575b4', '#91bfdb', '#fee090', '#fc8d59', '#d73027']
# cc = ['#4575b4', '#fc8d59', '#d73027', ]
# ltblue'#91bfdb', ltyellow'#fee090',

# multiRun and multiTime Colors
rtc= [u'#08306b', u'#083673', u'#083b7c', u'#084184', u'#08468c', u'#084c95',
      u'#09529d', u'#0d57a1', u'#115da5', u'#1562a9', u'#1a67ae', u'#1e6db2',
      u'#2272b6', u'#2878b9', u'#2e7ebc', u'#3383be',
      u'#3989c1', u'#3e8ec4', u'#4594c7', u'#4c99ca', u'#539dcc', u'#5aa2cf',
      u'#61a7d2', u'#68acd5', u'#6fb0d7', u'#78b5d9', u'#81badb', u'#89bfdd',
      u'#92c3de', u'#9bc8e0', u'#a2cce2', u'#a9cfe5', u'#b0d2e7', u'#b7d4ea',
      u'#bdd7ec', u'#c4daee', u'#c9ddf0', u'#cde0f1', u'#d1e2f3', u'#d5e5f4',
      u'#d9e8f5', u'#ddebf7', u'#e2edf8', u'#e6f0fa', u'#eaf3fb', u'#eef6fc',
      u'#f3f8fe', u'#f7fbff']  # Blues_r
# rtc = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99',
#        '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']
ls = [':', '-.', '--', '-']
lw = [1.0]*4  # [1.5, 1.0, 1.0, 0.5]
cycler = np.arange(len(ls)*len(rtc)).reshape(len(ls), len(rtc))

lm = ['o', 's', 'd', 'v', '<', '>', '^', 'p', 'h']
msize = np.array([2, 1.5, 2, 2, 2, 2, 2, 2, 2])*1.5

# Figures and Axes for histograms and spectra
fs1 = 8
fs2 = 8
fs3 = 6
lwid = 1.0
me = 512

fig = []
ax = []

# upper left
al, aw, ar = 0.55, 2.2, 0.05     # 2.75 in wide
ab, ah, at = 0.05, 1.8, 0.2      # 2.05 in tall
fw, fh = (aw+al+ar), (ah+ab+at)
axdim = [al/fw, ab/fh, aw/fw, ah/fh]

fig.append(plt.figure(1))
fig[0].clf()
fig[0].set_size_inches(fw, fh)
ax.append(fig[0].add_axes(axdim))

# upper right
al, aw, ar = 0.55, 2.2, 0.05      # 2.75 in wide
ab, ah, at = 0.05, 1.8, 0.2      # 2.05 in tall
fw, fh = (aw+al+ar), (ah+ab+at)
axdim = [al/fw, ab/fh, aw/fw, ah/fh]

fig.append(plt.figure(2))
fig[1].clf()
fig[1].set_size_inches(fw, fh)
ax.append(fig[1].add_axes(axdim))

# lower right
al, aw, ar = 0.55, 2.2, 0.05     # 2.75 in wide
ab, ah, at = 0.40, 1.8, 0.2     # 2.4 in tall
fw, fh = (aw+al+ar), (ah+ab+at)
axdim = [al/fw, ab/fh, aw/fw, ah/fh]

fig.append(plt.figure(3))
fig[2].clf()
fig[2].set_size_inches(fw, fh)
ax.append(fig[2].add_axes(axdim))

# lower left
al, aw, ar = 0.55, 1.8, 0.45     # 2.75 in wide
ab, ah, at = 0.40, 1.8, 0.2      # 2.4 in tall
fw, fh = (aw+al+ar), (ah+ab+at)
axdim = [al/fw, ab/fh, aw/fw, ah/fh]

fig.append(plt.figure(4))
fig[3].clf()
fig[3].set_size_inches(fw, fh)
ax.append(fig[3].add_axes(axdim))

# al, aw, ar = 0.25, 1.4, 0.55     # 2.20 in wide
# ab, ah, at = 0.35, 1.4, 0.2      # 1.95 in tall
# fw, fh = (aw+al+ar), (ah+ab+at)
# axdim = [al/fw, ab/fh, aw/fw, ah/fh]

# fig.append(plt.figure(1))
# fig[0].clf()
# fig[0].set_size_inches(fw, fh)
# ax.append(fig[0].add_axes(axdim))

# al, aw, ar = 0.25, 1.4, 0.55     # 2.20 in wide
# ab, ah, at = 0.40, 1.4, 0.2      # 2.00 in tall
# fw, fh = (aw+al+ar), (ah+ab+at)
# axdim = [al/fw, ab/fh, aw/fw, ah/fh]

# fig.append(plt.figure(2))
# fig[1].clf()
# fig[1].set_size_inches(fw, fh)
# ax.append(fig[1].add_axes(axdim))

# Figure and Axes for 2D slice images
sf = 1.0
pfnt = 8
lfnt = 8

al, aw, ar = 0.35, 1.4, 0.10     # 1.85 in wide
ab, ah, at = 0.35, 1.4, 0.20     # 1.95 in tall
fw, fh = (aw+al+ar)*sf, (ah+ab+at)*sf
axdim = [al*sf/fw, ab*sf/fh, aw*sf/fw, ah*sf/fh]

fig1 = plt.figure(0)
fig1.clf()
fig1.set_size_inches(fw, fh)
ax1 = fig1.add_axes(axdim)

al, aw, ar = 0.55, 0.55, 0.75     # 1.85 in wide
ab, ah, at = 1.55, 0.08, 0.32     # 1.95 in tall
fw, fh = (aw+al+ar)*sf, (ah+ab+at)*sf
axdim = [al*sf/fw, ab*sf/fh, aw*sf/fw, ah*sf/fh]

cax0 = fig1.add_axes(axdim)

al, aw, ar = 1.05, 0.55, 0.25     # 1.85 in wide
ab, ah, at = 1.55, 0.08, 0.32     # 1.95 in tall
fw, fh = (aw+al+ar)*sf, (ah+ab+at)*sf
axdim = [al*sf/fw, ab*sf/fh, aw*sf/fw, ah*sf/fh]

cax1 = fig1.add_axes(axdim)

al, aw, ar = 1.05, 0.55, 0.25     # 1.85 in wide
ab, ah, at = 0.60, 0.08, 1.27     # 1.95 in tall
fw, fh = (aw+al+ar)*sf, (ah+ab+at)*sf
axdim = [al*sf/fw, ab*sf/fh, aw*sf/fw, ah*sf/fh]

cax2 = fig1.add_axes(axdim)

al, aw, ar = 0.50, 0.55, 0.80     # 1.85 in wide
ab, ah, at = 0.60, 0.08, 1.27     # 1.95 in tall
fw, fh = (aw+al+ar)*sf, (ah+ab+at)*sf
axdim = [al*sf/fw, ab*sf/fh, aw*sf/fw, ah*sf/fh]

cax3 = fig1.add_axes(axdim)

cax0.set_axis_off()
cax1.set_axis_off()
cax2.set_axis_off()
cax3.set_axis_off()


def blankSpace(val, pos):
    return " "


def linFormatter(val, pos):
    if val >= 0:
        string = '$%d$' % int(val)
    else:
        string = '$\\text{-}%d$' % int(abs(val))
    return string


def logFormatter(val, pos):
    exp = int(np.log10(val))
    if exp > 0:
        string = '$10^{{{}}}$'.format(exp)
    elif exp == 0:
        string = '$10^0$'
    else:
        string = '$10^{{\\text{{-}}{}}}$'.format(abs(exp))
    return string


def rsfig(x, sigfigs):
    """
    Rounds the value(s) in x to the number of significant figures in sigfigs.

    Restrictions:
    sigfigs must be an integer type and store a positive value.
    x must be a real value or an array like object containing only real values.
    """
    if not (type(sigfigs) is int or np.issubdtype(sigfigs, np.integer)):
        raise TypeError("rsfig: sigfigs must be an integer.")

    if not np.all(np.isreal(x)):
        raise TypeError("rsfig: all x must be real.")

    if sigfigs <= 0:
        raise ValueError("rsfig: sigfigs must be positive.")

    mantissas, binaryExponents = np.frexp(x)

    decimalExponents = log10(2) * binaryExponents
    intParts = np.floor(decimalExponents)

    mantissas *= 10.0**(decimalExponents - intParts)

    return np.around(mantissas, decimals=sigfigs - 1) * 10.0**intParts


def imagesc(var, cm, title, filename, cmax=None, normed=None):
    if cmax:
        try:
            img = ax1.imshow(var, cmap=cm, aspect=1,
                             vmin=cmax[0], vmax=cmax[1])
        except:
            img = ax1.imshow(var, cmap=cm, aspect=1, vmin=-cmax, vmax=cmax)
    elif normed:
        img = ax1.imshow(var, cmap=cm, aspect=1, vmin=0, vmax=1)
    else:
        img = ax1.imshow(var, cmap=cm, aspect=1)

    ax1.set_title(title, fontsize=lfnt)
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    cb = fig1.colorbar(img, cax=cax, ax=ax1)
    cb.locator = MaxNLocator(nbins=6)
    cb.formatter.set_useOffset(False)
    cb.formatter.set_powerlimits((-4, 4))
    cb.ax.tick_params(labelsize=pfnt)
    cb.ax.yaxis.get_offset_text().set_size(pfnt)
    cb.ax.yaxis.set_offset_position('left')
    cb.update_ticks()
    fig1.savefig(filename)
    ax1.cla()
    cax.cla()


def image_2d_hist(hist, xyrange, Xf, Yf, cm, xlab, ylab, title, filename,
                  cpos=1, cmax=None, normed=None):
    if cmax:
        img = ax1.imshow(hist, cmap=cm, interpolation='bilinear',
                         origin='low', extent=xyrange,
                         vmin=cmax[0], vmax=cmax[1])
    elif normed:
        img = ax1.imshow(hist, cmap=cm, interpolation='bilinear',
                         origin='low', extent=xyrange, vmin=0, vmax=1)
    else:
        img = ax1.imshow(hist, cmap=cm, interpolation='bilinear',
                         origin='low', extent=xyrange)
    ax1.hold(True)
    ax1.plot(Xf, Yf, c='#2166ac', lw=lwid)
    # ax1.plot(Xf, Yf, 'w:', lw=lwid)

    ax1.axis('tight')
    # ax1.set_title(title, fontsize=fs2)
    ax1.set_xlabel(xlab, fontsize=fs2)
    ax1.set_ylabel(ylab, fontsize=fs2)
    ax1.set_title(title, fontsize=fs2, x=0.05, y=0.85, loc='left')

    ax1.tick_params(labelsize=fs1)
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3))
    ax1.xaxis.get_offset_text().set_size(fs1)
    ax1.yaxis.get_offset_text().set_size(fs1)
    ax1.yaxis.set_minor_formatter(FuncFormatter(blankSpace))

    if cpos == 0:
        cax = cax0
    elif cpos == 1:
        cax = cax1
    elif cpos == 2:
        cax = cax2
    else:
        cax = cax3

    cax.set_axis_on()
    cb = fig1.colorbar(img, cax=cax, ax=ax1, orientation='horizontal')
    cb.locator = MaxNLocator(nbins=4)
    cb.formatter.set_useOffset(False)
    cb.formatter.set_powerlimits((-4, 4))

    cb.ax.tick_params(labelsize=fs3)
    cb.ax.yaxis.get_offset_text().set_size(fs1)
    cb.ax.yaxis.set_offset_position('left')
    cb.update_ticks()

    fig1.savefig(filename)
    ax1.cla()
    cax.cla()
    cax.set_axis_off()

##############################################################################


def load_histogram(fname):
    with open(fname) as fh:
        fh.readline()
        fh.readline()
        m1 = float(fh.readline())
        m2 = float(fh.readline())
        ens_min = float(fh.readline())
        ens_max = float(fh.readline())
        ens_width = float(fh.readline())
        nbins = int(fh.readline())

    hist = np.loadtxt(fname, skiprows=8)

    try:
        hist *= sqrt(m2-m1**2)/ens_width
    except:
        print('broken normalization')
        print(case, ttag, rtag, hist_tag)

    cntrs = np.linspace(ens_min+0.5*ens_width,
                        ens_max-0.5*ens_width, nbins)

    return hist, m1, m2, cntrs


def plot_multirun_histogram(data_dir, fig_dir, case, ttag, rtags, hist_tag,
                            xlab=None, title=None):
    ax[3].cla()
    nr = len(rtags)

    for k in range(nr):
        hist, m1, m2, cntrs = load_histogram(
                                data_dir, case, ttag, rtags[k], hist_tag)

        kl, kc = np.argwhere(cycler == k % cycler.size)[0]
        ax[3].semilogy(cntrs, hist, label=rtags[k],
                       c=rtc[kc], ls='-', lw=lwid)
        ax[3].hold(True)

    lims = ax[3].axis('tight')
    ax[3].hold(False)

    # Ensemble-averaged histograms figure without normalization
    width = lims[1] - lims[0]
    ax[3].axis([lims[0]-0.1*width, lims[1]+0.1*width,
                lims[2]/2, lims[3]*2])

    ax[3].set_ylabel('PDF', fontsize=fs2)
    if xlab:
        ax[3].set_xlabel(xlab, fontsize=fs1)
    if title:
        ax[3].set_title(title, fontsize=fs2, x=0.01, y=0.98, loc='left')

    ax[3].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,
                 fontsize=fs1, handlelength=2, frameon=False,
                 handletextpad=0.1)

    ax[3].tick_params(labelsize=fs3)
    ax[3].tick_params(axis='y', which='major', pad=1)
    ax[3].yaxis.set_major_formatter(FuncFormatter(logFormatter))
    ax[3].yaxis.set_minor_formatter(FuncFormatter(blankSpace))

    ax[3].grid(True, ls=':', linewidth=0.5, color='w', alpha=0.2)
    ax[3].set_axisbelow(True)

    fig[3].savefig('%spdfs/multirun/%s-%4.4d_%s.png'
                   % (fig_dir, case, ttag, hist_tag))


def plot_tseries_histogram(data_dir, fig_dir, prefix, ttags, suffix,
                           xlab=None, title=None):
    ax[3].cla()
    nr = len(ttags)

    for k in range(nr):
        fname = '%s/%s-%s_%s.hist' % (data_dir, prefix, ttags[k], suffix)
        hist, m1, m2, cntrs = load_histogram(fname)

        kl, kc = np.argwhere(cycler == k % cycler.size)[0]
        ax[3].semilogy(cntrs, hist, label=ttags[k],
                       c=rtc[kc], ls='-', lw=lwid)
        ax[3].hold(True)

    lims = ax[3].axis('tight')
    ax[3].hold(False)

    # Ensemble-averaged histograms figure without normalization
    width = lims[1] - lims[0]
    ax[3].axis([lims[0]-0.1*width, lims[1]+0.1*width,
                lims[2]/2, lims[3]*2])

    ax[3].set_ylabel('PDF', fontsize=fs2)
    if xlab:
        ax[3].set_xlabel(xlab, fontsize=fs1)
    if title:
        ax[3].set_title(title, fontsize=fs2, x=0.01, y=0.98, loc='left')

    ax[3].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,
                 fontsize=fs1, handlelength=2, frameon=False,
                 handletextpad=0.1)

    ax[3].tick_params(labelsize=fs3)
    ax[3].tick_params(axis='y', which='major', pad=1)
    ax[3].yaxis.set_major_formatter(FuncFormatter(logFormatter))
    ax[3].yaxis.set_minor_formatter(FuncFormatter(blankSpace))

    ax[3].grid(True, ls=':', linewidth=0.5, color='w', alpha=0.2)
    ax[3].set_axisbelow(True)

    fig[3].savefig('%s/pdfs/tseries/%s-%s.png' % (fig_dir, prefix, suffix))


def plot_tseries_spectra(data_dir, fig_dir, prefix, ttags, suffix, nk,
                         title=None):
    ax[3].cla()
    nt = len(ttags)

    for k in range(nt):
        fname = '%s/%s-%s_%s.spectra' % (data_dir, prefix, ttags[k], suffix)

        with open(fname) as fh:
            fh.readline()
            sylab = fh.readline().rstrip('\n')

        spect = np.loadtxt(fname, skiprows=2)

        ki = np.arange(1, nk, dtype=np.float64)

        try:
            spect[1:] *= (ki)**(5./3)

            # kl, kc = np.argwhere(cycler == k % cycler.size)[0]
            # kl = k // (len(rtc)//len(ls))
            kc = k % len(rtc)
            # print kl, kc
            ax[3].loglog(ki, spect[1:], label=ttags[k],
                         c=rtc[kc], ls='-', lw=0.75*lwid)
            if k == nt-1:
                ax[3].loglog(ki, spect[1:], 'm-', label=ttags[k], lw=1.5*lwid)

        except:
            continue

    try:
        lims = ax[3].axis('tight')
        # ax[3].set_ylim([0.3*max(spect[nk//2], lims[2]), lims[3]*1.5])
        # ax[3].set_xlim([lims[0]/2, lims[1]*2])
        # ax[3].plot([lims[0]/2, lims[1]*2], [1.6, 1.6], 'w:')
    except:
        pass

    ax[3].grid(True, ls=':', linewidth=0.5, color='w', alpha=0.2)
    ax[3].set_axisbelow(True)
    ax[3].tick_params(labelsize=fs1)

    ax[3].set_xlabel('$k$', fontsize=fs2)
    # ax[3].set_ylabel('$(k^{5/3})%s$'
    #                  % sylab, fontsize=fs2)  # \epsilon^{-2/3}

    if title:
        ax[3].set_title(title, fontsize=fs2, x=0.01, y=1.00, loc='left')
    else:
        ax[3].set_title('PSD time series', fontsize=fs2, x=0.01, y=1.00,
                        loc='left')

    ax[3].legend(bbox_to_anchor=(1.01, 1.05), loc=2, borderaxespad=0.,
                 fontsize=2.5, handlelength=4, frameon=False,
                 handletextpad=0.1)

    fig[3].savefig('%s/spectra/tseries/%s-%s.png' % (fig_dir, prefix, suffix))

##############################################################################
