import matplotlib.pyplot as plt
import numpy as np

cmaps = {}

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

colormap = 'rainbow'
cmap_list = [colormap]

# Create figure and adjust figure height to number of colormaps
nrows = 1
figh = 0.8
fig, ax = plt.subplots(nrows=1, figsize=(8, figh))
dh = 0.04
fig.subplots_adjust(top=0.4, bottom=0.15, left=0.05 + dh, right=0.95 - dh)
# ax.set_title('$M\\left(v_{z,0},v_{\\perp,0}\\right)$', fontsize=14)
# ax.set_title('$max(\\Delta v)$', fontsize=14)
ax.set_title('$\\mathrm{max}(\\Delta v)$', fontsize=14)
# ax.set_title('$\\mathbf{max(\\Delta v)}$', fontsize=14)


ax.imshow(gradient, aspect='auto', cmap=colormap)
# ax.text(-0.02, 0.5, '0', va='center', ha='right', fontsize=13, transform=ax.transAxes)
# ax.text(1.02, 0.5, '1', va='center', ha='left', fontsize=13, transform=ax.transAxes)
ax.text(-0.01, 0.5, '0', va='center', ha='right', fontsize=14, transform=ax.transAxes)
ax.text(1.01, 0.5, '$2v_{th,T}$', va='center', ha='left', fontsize=14, transform=ax.transAxes)

ax.set_axis_off()

## save plots to file
# save_fig_dir = '../../../Papers/texts/paper2022/pics/'
# file_name = 'v_space_metric_colorbar'
# beingsaved = plt.gcf()
# beingsaved.savefig(save_fig_dir + file_name + '.eps', format='eps')
