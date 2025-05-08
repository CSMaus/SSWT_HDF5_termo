import h5py
import numpy as np
from ssqueezepy import ssq_cwt, issq_cwt, extract_ridges
from ssqueezepy.visuals import imshow, plot, scat
from ssqueezepy.toolkit import lin_band
from ssqueezepy.visuals import imshow, plot
import matplotlib.pyplot as plt
import cv2
import numpy as np

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
threshold_percent = 0.07
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
# sorted_indices = np.argsort(-areas)[:5]
sorted_indices = sorted(np.argsort(-areas)[:5], key=lambda idx: stats[idx + 1, cv2.CC_STAT_LEFT])

plt.figure(figsize=(6, 6))
plt.imshow(mask, aspect='auto', origin='lower', cmap='gray')
for i, idx in enumerate(sorted_indices):
    x, y, w, h, area = stats[idx + 1]
    max_x, min_x = x + w, x
    max_y, min_y = y + h, y
    print(f'Object {i + 1}: minX={min_x}, maxX={max_x}, minY={min_y}, maxY={max_y}')
    plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2))
    plt.text(x, y - 5, str(i + 1), color='yellow', fontsize=12, weight='bold')

plt.title(f'Mask > {threshold_percent*100:.0f}%')
plt.title(f'Tx_{i}')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()

bbox_info = []
for idx in sorted_indices:
    x, y, w, h = stats[idx + 1, cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT + 1]
    bbox_info.append({'idx': idx, 'x': x, 'y': y, 'w': w, 'h': h, 'min_y': y, 'max_y': y + h})
merged = []
reference = bbox_info[0]
merged.append(reference)
for b in bbox_info[1:]:
    if not (b['max_y'] < reference['min_y'] or b['min_y'] > reference['max_y']):
        merged.append(b)
min_x = min(b['x'] for b in merged)
max_x = max(b['x'] + b['w'] for b in merged)
min_y = min(b['y'] for b in merged)
max_y = max(b['y'] + b['h'] for b in merged)
w = max_x - min_x
h = max_y - min_y

plt.figure(figsize=(6, 6))
plt.imshow(mask, aspect='auto', origin='lower', cmap='gray')
plt.gca().add_patch(plt.Rectangle((min_x, min_y), w, h, edgecolor='red', facecolor='none', linewidth=2))
plt.text(min_x, min_y - 5, 'merged', color='yellow', fontsize=12, weight='bold')
plt.title(f'Merged bbox')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()
bbox_mask = np.zeros_like(Tx, dtype=bool)
bbox_mask[min_y:max_y, min_x:max_x] = True
refined_mask = np.logical_and(bbox_mask, mask)
Tx_merged = np.where(refined_mask, Tx, 0 + 0j)

# unite bbox from last code
num_bbox_to_use = 2
if num_bbox_to_use < 2:
    i = 0
    idx = sorted_indices[i]
    x, y, w, h, area = stats[idx + 1]
    bbox_mask = np.zeros_like(Tx, dtype=bool)
    bbox_mask[y:y+h, x:x+w] = True

    Tx_i = np.where(bbox_mask, Tx, 0+0j)
    imshow(Tx_i)
else:
    bbox_mask = np.zeros_like(Tx, dtype=bool)
    for i in range(num_bbox_to_use):
        idx = sorted_indices[i]
        x, y, w, h, area = stats[idx + 1]
        bbox_mask[y:y + h, x:x + w] = True

    Tx_i = np.where(bbox_mask, Tx, 0 + 0j)



xrec = issq_cwt(Tx_merged, wavelet=('morlet', {'mu': 4.5}))
plt.plot(xrec)
plt.show()

freqs = np.fft.rfftfreq(len(xrec)*1000, dt)
fft = np.abs(np.fft.rfft(xrec, n=len(freqs)*2-1))
plt.plot(freqs, fft)
plt.show()

