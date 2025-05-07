import h5py
import numpy as np
from ssqueezepy import ssq_cwt, issq_cwt, extract_ridges
from ssqueezepy.visuals import imshow, plot, scat
from ssqueezepy.toolkit import lin_band
from ssqueezepy.visuals import imshow, plot
import matplotlib.pyplot as plt
import cv2

# Load the signal data from an HDF5 file
import os
hdf5_files = [f for f in os.listdir('.') if f.endswith('.hdf5')]

dfile_name = hdf5_files[2]  # '300_21.hdf5'
print("Data file is", dfile_name)
with h5py.File(dfile_name, 'r') as f:
    time = f['Time/x'][:f['Time/x'].shape[0]//5]
    data = f['Time/Point_1/Meas_In/Average'][:f['Time/Point_1/Meas_In/Average'].shape[0]//5]

# Compute the Synchrosqueezed CWT
kw = dict(wavelet=('morlet', {'mu': 4.5}), nv=16, scales='log')
Tx, Wx, ssq_freqs, scales = ssq_cwt(data, t=time, **kw)

#%%# Estimate inversion ridge ###############################################
bw, slope, offset = 0.03, -0.0001, -3100
Cs, freqband = lin_band(Tx, slope, offset, bw, norm=(0, 1e-3))
print(Cs)
print(freqband)
# TODO: the task is to define Cs and freqband automatically (not using lin_band)

#%%###########################################################################
morlet_mu = 4.5
kw = dict(wavelet=('morlet', {'mu': morlet_mu}), nv=32, scales='log')
xrec = issq_cwt(Tx, kw['wavelet'], Cs, freqband)[0]

dt = time[1]
freqs = np.fft.rfftfreq(len(time[:2300]), dt)
fft = np.abs(np.fft.rfft(xrec[:2300]))

# fig, ax = plt.subplots(1, 2, figsize=(10, 8))
#ax[0].plot(time[:2300],xrec[:2300])
#ax[1].plot(freqs,fft)
#ax[1].set_xlim(0,1e6)

# _____________________________ NEW CODE ___________________________
threshold_percent = 0.1
threshold_value = threshold_percent * np.max(np.abs(Tx))
print("np.max(np.abs(Tx)", np.max(np.abs(Tx)))
print("Thresh val:", threshold_value)
mask = np.abs(Tx) >= threshold_value

# automatically define the band ans freqs
ys, xs = np.where(mask)
min_x, max_x = xs.min(), xs.max()
min_y, max_y = ys.min(), ys.max()
print(f"X lims: {min_x} : {max_x}")
print(f"Y lims: {min_y} : {max_y}")


mask_uint8 = mask.astype(np.uint8)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
areas = stats[1:, cv2.CC_STAT_AREA]
sorted_indices = np.argsort(-areas)[:2]

plt.figure(figsize=(6, 6))
# plt.imshow(mask, aspect='auto', origin='lower', cmap='gray')
for i, idx in enumerate(sorted_indices):
    x, y, w, h, area = stats[idx + 1]
    max_x, min_x = x + w, x
    max_y, min_y = y + h, y
    print(f'Object {i + 1}: minX={min_x}, maxX={max_x}, minY={min_y}, maxY={max_y}')
    # plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2))
    # plt.text(x, y - 5, str(i + 1), color='yellow', fontsize=12, weight='bold')

i = 0
idx = sorted_indices[i]
x, y, w, h, area = stats[idx + 1]
bbox_mask = np.zeros_like(Tx, dtype=bool)
bbox_mask[y:y+h, x:x+w] = True

Tx_i = np.where(bbox_mask, Tx, 0+0j)
imshow(Tx_i)
# plt.title(f'Mask > {threshold_percent*100:.0f}%')
# plt.title(f'Tx_{i}')
# plt.tight_layout()
# plt.gca().invert_yaxis()
# plt.show()

xrec = issq_cwt(Tx_i, wavelet=('morlet', {'mu': 4.5}))
plt.plot(xrec)
plt.show()

