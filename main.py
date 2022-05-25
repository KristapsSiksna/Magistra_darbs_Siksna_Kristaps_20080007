import profile

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import cv2
from cProfile import Profile
from pstats import Stats

prof = Profile()
prof.enable()

# ----------------------Image files------------------------------

#image = "FITS_images/P039+65_fullband-MFS-I-image-pb.fits"
image = "FITS_images/P045+69_fullband-MFS-I-image-pb.fits"
#image = "FITS_images/P048+65_fullband-MFS-I-image-pb.fits"

# ---------------------------Template sample for sample matching-------------

#template_image_045 = 'FITS_images/Templates/P045_P048_template_1.fits'
#template_image_045 = 'FITS_images/Templates/P045_P048_template_2.fits'
template_image_045 = 'FITS_images/Templates/P045_P048_template_3.fits'

template_file = fits.open(template_image_045)
template_data = template_file[0].data

# --------------------------------------Image data------------------------
image_file = fits.open(image)
# image data
image_data = image_file[0].data
# image header
image_header = image_file[0].header
# ++++++++++++++++++++++++initialize wcs++++++++++++++++++++++++++++++++++++++++++++++++++
wcs = WCS(header=image_header, fix=True)[0, 0, :, :]

# ++++++++++++++++++++++++image header++++++++++++++++++++++++++++++++++++++++++++++++++
# print('\n Header:')
# print(repr(image_file[0].header))

# plt.subplot(121)
# plt.title('before')
# plt.imshow(template_data)

# remove nan values
nn = np.argwhere(~np.isnan(template_data))
template_data = template_data[np.min(nn[0]):nn[-1][0], nn[0][-1]:np.max(nn[:-1])]
#
# plt.subplot(122, projection=wcs)
# plt.title('after')
# plt.imshow(template_data, cmap='cool')
# plt.show()
# exit()

# plot
top = 0.95
bottom = 0.1
left = 0.068
right = 0.988
hspace = 0.5
wspace = 0.2

# plot image
fig = plt.figure(figsize=(16, 8), dpi=100)
fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
ax = fig.add_subplot(121, projection=wcs)
ax2 = fig.add_subplot(122)
# plot in log scale
# data_abs = np.abs(image_data[0, 0, :, :])
# data_log = np.log10(data_abs) * 10

ax.set_title('Oriģinālais attēls')
ax.set_xlabel('Rektascensija  [h:m:s]')
ax.set_ylabel('Deklinācija [deg]')
im = ax.imshow(image_data[0, 0, :, :], cmap='gray')
# add colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label('[Jy / Beam]')

# histogram
NBINS = 1000

y1, x1, _ = ax2.hist(image_data[0, 0, :, :].flatten(), NBINS)
ax2.set_yscale('log')
ax2.set_title('Oriģinālā attēla histogramma')
ax2.set_xlabel('Pikseļu intensitāte [Jy/Beam]')
ax2.set_ylabel('Pikseļu skaits')

# plot template
fig = plt.figure(figsize=(16, 8), dpi=100)
fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
ax = fig.add_subplot(121, projection=wcs)
ax2 = fig.add_subplot(122)
# plot in log scale
# data_abs = np.abs(template_data)
# data_log = np.log10(data_abs) * 10

ax.set_title('Oriģinālais parauga attēls')
ax.set_xlabel('Rektascensija [h:m:s]')
ax.set_ylabel('Deklinācija [deg]')
im = ax.imshow(template_data, cmap='gray')
# add colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label('[Jy / Beam]')

# histogram
NBINS = 1000

y2, x2, _ = ax2.hist(template_data.flatten(), NBINS)
ax2.set_yscale('log')
ax2.set_title('Parauga histogramma')
ax2.set_xlabel('Pikseļu intensitāte [Jy/Beam]')
ax2.set_ylabel('Pikseļu skaits')

# -------------------------Dynamic range and RMS-------------------------------------
def rmsValue(arr):
    square = 0
    mean = 0.0
    root = 0.0
    # Calculate square
    square = arr ** 2
    # Calculate Mean
    mean = np.mean(square)
    # Calculate Root
    root = np.sqrt(mean)
    return root


def signalVsNoise(im):
    return np.abs(np.median(im)) / rmsValue(im)


RMS_image = rmsValue(image_data)
print('RMS image: ', RMS_image)
dynamicRangeImage = signalVsNoise(image_data)
image_std = np.std(image_data)
print('std image: ', image_std)

RMS_template = rmsValue(template_data)
print('RMS template: ', RMS_template)
dynamicRangeTemplate = signalVsNoise(template_data)
print('Dynamic range template: ', dynamicRangeTemplate)
template_std = np.std(template_data)
print('std template: ', template_std)

# -------------------------image rms and signal/noise max intensity-------------------------------------
x1_max = x1[np.where(y1 == y1.max())]

image_rms_values = []
image_signal_vs_noise_values_median_intensity = []
image_stds = []
image = image_data[0, 0, :, :]

for i in np.linspace(0, 10, 100):
    image_stds.append(i)
    # calc max and min intensity for intervals
    max_intensity = x1_max + image_std * i
    min_intensity = x1_max - image_std * i

    # make intervals
    image_interval_i = np.clip(image, min_intensity, max_intensity)

    # calculate rms
    rms_i_value = rmsValue(image_interval_i)
    image_rms_values.append(rms_i_value)
    # calc median
    median_i_image = np.median(image_interval_i)

    # calculate signal/noise median
    signal_vs_noise_value_median = np.abs(median_i_image) / rms_i_value
    image_signal_vs_noise_values_median_intensity.append(signal_vs_noise_value_median)

image_rms_values_max = np.asarray(image_rms_values)
image_signal_vs_noise_values_median = np.asarray(image_signal_vs_noise_values_median_intensity)
image_stds = np.asarray(image_stds)

std_where_median_value_image = image_stds[np.argmax(image_signal_vs_noise_values_median)]

# -----------plot---------
top = 0.95
bottom = 0.6
left = 0.068
right = 0.988
hspace = 0.5
wspace = 0.2
fig = plt.figure(figsize=(16, 8), dpi=100)
fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax.plot(np.linspace(0, 10, 100), image_rms_values_max, color='red')
#ax.scatter(np.linspace(0, 10, 100), image_rms_values_mean, label='mean intensity', color='green')
#ax.scatter(np.linspace(0, 10, 100), image_rms_values_median, label='median intensity', color='blue')
#ax.legend(loc='lower right')
ax.set_title('Attēla intervālu RMS vērtības')
ax.set_xlabel('Standartnovirze')
ax.set_ylabel('RMS vērtība')
ax2.plot(np.linspace(0, 10, 100), image_signal_vs_noise_values_median, color='purple')
#ax2.legend(loc='lower right')
ax2.set_title('Attēla signāla / troksni attiecība \n (mediānas vērtība/RMS)')
ax2.set_xlabel('Standartnovirze')
ax2.set_ylabel('Signāla / troksni attiecība')
fig.tight_layout()

#make best image
#doesn`t work because max is at 0 std`s
#best_image_median_intensity = np.clip(image, x1_max - std_where_median_value_image * image_std,
                                      x1_max + std_where_median_value_image * image_std)
#make best image using 10 std`s
best_image_median_intensity = np.clip(image, x1_max - 10 * image_std, x1_max + 10 * image_std)

print('max signal vs noise median intensity image: ', np.max(best_image_median_intensity))

fig = plt.figure()
ax = plt.subplot(111, projection=wcs)
ax.set_axis_off()
ax.imshow(best_image_median_intensity, cmap='gray')
plt.savefig('best_image.jpg')

# ---------------------------template rms and signal/noise-------------------------------
x2_max = x2[np.where(y2 == y2.max())]

template_rms_values = []
template_signal_vs_noise_values_median_intensity = []
template_stds = []

for i in np.linspace(0, 10, 100):
    template_stds.append(i)
    # calc max and min intensity
    max_intensity = x2_max + template_std * i
    min_intensity = x2_max - template_std * i

    # make intervals
    template_interval_i = np.clip(template_data, min_intensity, max_intensity)

    # calculate rms
    rms_i_value = rmsValue(template_interval_i)
    template_rms_values.append(rms_i_value)

    # median
    median_i_template = np.median(template_interval_i)

    # calculate signal/noise median
    signal_vs_noise_value_median = np.abs(median_i_template) / rms_i_value
    template_signal_vs_noise_values_median_intensity.append(signal_vs_noise_value_median)

template_rms_values = np.asarray(template_rms_values)
template_signal_vs_noise_values_median = np.asarray(template_signal_vs_noise_values_median_intensity)
template_stds = np.asarray(template_stds)

std_where_median_value_template = template_stds[np.argmax(template_signal_vs_noise_values_median)]

# -----------plot---------
top = 0.95
bottom = 0.6
left = 0.068
right = 0.988
hspace = 0.5
wspace = 0.2
fig = plt.figure(figsize=(16, 8), dpi=100)
fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax.plot(np.linspace(0, 10, 100), template_rms_values, color='red')
#ax.scatter(np.linspace(0, 10, 100), template_rms_values_mean, label='mean intensity', color='green')
#ax.scatter(np.linspace(0, 10, 100), template_rms_values_median, label='median intensity', color='blue')
#ax.legend(loc='lower right')
ax.set_title('Parauga intervālu RMS vērtības')
ax.set_xlabel('Standartnovirze')
ax.set_ylabel('RMS vērtība')
ax2.plot(np.linspace(0, 10, 100), template_signal_vs_noise_values_median, color='purple')
#ax2.legend(loc='lower right')
ax2.set_title('Parauga signāla / troksni attiecība \n (mediānas vērtība/RMS)')
ax2.set_xlabel('Standartnovirze')
ax2.set_ylabel('Signāla / troksni attiecība')
fig.tight_layout()

#make best template image
best_template_median_intensity = np.clip(template_data, x2_max - 10 * template_std,
                                      x2_max + 10 * template_std)

print('max signal vs noise median intensity template: ', np.max(best_template_median_intensity))

# -------------------------Image smothing and sharpening-------------------------------------
#make image for sharpening, smoothing etc...
best_image = np.clip(image_data[0,0,:,:], -0.1, 0.1)

# sharpening
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
print('sharpen kernel:', kernel)
image_sharp = cv2.filter2D(src=best_image, ddepth=-1, kernel=kernel)
print("RMS of sharpen image: ", rmsValue(image_sharp))
print("Signal vs noise sharpen image: ", signalVsNoise(image_sharp))
template_sharp = cv2.filter2D(src=best_template_median_intensity, ddepth=-1, kernel=kernel)
print("\nRMS of sharpen template: ", rmsValue(template_sharp))
print("Signal vs noise sharpen template: ", signalVsNoise(template_sharp))

# Gaussian blur image
Gaussian_blur_image_1 = cv2.GaussianBlur(best_image, (3, 3), 0)
Gaussian_blur_image_2 = cv2.GaussianBlur(best_image, (5, 5), 0)
Gaussian_blur_image_3 = cv2.GaussianBlur(best_image, (3, 3), 10)
Gaussian_blur_image_4 = cv2.GaussianBlur(best_image, (5, 5), 10)

print("\nRMS of Gaussian image 1 (kernel 3x3, sigma 0): ", rmsValue(Gaussian_blur_image_1))
print("RMS of Gaussian image 2 (kernel 5x5, sigma 0): ", rmsValue(Gaussian_blur_image_2))
print("RMS of Gaussian image 3 (kernel 3x3, sigma 10): ", rmsValue(Gaussian_blur_image_3))
print("RMS of Gaussian image 4 (kernel 5x5, sigma 10): ", rmsValue(Gaussian_blur_image_4))

print("\nSignal vs noise Gaussian image 1 (kernel 3x3, sigma 0): ", signalVsNoise(Gaussian_blur_image_1))
print("Signal vs noise Gaussian image 2 (kernel 5x5, sigma 0): ", signalVsNoise(Gaussian_blur_image_2))
print("Signal vs noise Gaussian image 3 (kernel 3x3, sigma 10): ", signalVsNoise(Gaussian_blur_image_3))
print("Signal vs noise Gaussian image 4 (kernel 5x5, sigma 10): ", signalVsNoise(Gaussian_blur_image_4))

# Gaussian blur template
Gaussian_blur_template_1 = cv2.GaussianBlur(best_template_median_intensity, (3, 3), 0)
Gaussian_blur_template_2 = cv2.GaussianBlur(best_template_median_intensity, (5, 5), 0)
Gaussian_blur_template_3 = cv2.GaussianBlur(best_template_median_intensity, (3, 3), 10)
Gaussian_blur_template_4 = cv2.GaussianBlur(best_template_median_intensity, (5, 5), 10)

print("\nRMS of Gaussian template 1 (kernel 3x3, sigma 0): ", rmsValue(Gaussian_blur_template_1))
print("RMS of Gaussian template 2 (kernel 5x5, sigma 0): ", rmsValue(Gaussian_blur_template_2))
print("RMS of Gaussian template 3 (kernel 3x3, sigma 10): ", rmsValue(Gaussian_blur_template_3))
print("RMS of Gaussian template 4(kernel 5x5, sigma 10): ", rmsValue(Gaussian_blur_template_4))

print("\nSignal vs noise Gaussian template 1 (kernel 3x3, sigma 0): ", signalVsNoise(Gaussian_blur_template_1))
print("Signal vs noise Gaussian template 2 (kernel 5x5, sigma 0): ", signalVsNoise(Gaussian_blur_template_2))
print("Signal vs noise Gaussian template 3 (kernel 3x3, sigma 10): ", signalVsNoise(Gaussian_blur_template_3))
print("Signal vs noise Gaussian template 4 (kernel 5x5, sigma 10): ", signalVsNoise(Gaussian_blur_template_4))

# Median blur image
Median_blur_image_1 = cv2.medianBlur(best_image, 3)
Median_blur_image_2 = cv2.medianBlur(best_image, 5)

print("\nRMS of Median blur image 1(kernel size 3): ", rmsValue(Median_blur_image_1))
print("RMS of Median blur image 2(kernel size 4): ", rmsValue(Median_blur_image_2))

print("\nSignal vs noise Median blur image 1 (kernel size 3): ", signalVsNoise(Median_blur_image_1))
print("Signal vs noise Median blur image 2 (kernel size 5): ", signalVsNoise(Median_blur_image_2))

# Median blur template
Median_blur_template_1 = cv2.medianBlur(best_template_median_intensity, 3)
Median_blur_template_2 = cv2.medianBlur(best_template_median_intensity, 5)

print("\nRMS of Median blur template 1 (kernel size 3): ", rmsValue(Median_blur_template_1))
print("RMS of Median blur template 2 (kernel size 5): ", rmsValue(Median_blur_template_2))

print("\nSignal vs noise Median blur image 1 (kernel size 3): ", signalVsNoise(Median_blur_template_1))
print("Signal vs noise Median blur image 2 (kernel size 5): ", signalVsNoise(Median_blur_template_2))

# plot image and template sharpen

fig = plt.figure(figsize=(16, 8), dpi=100)
# fig.subplots_adjust(top=0.956, bottom=0.074, left=0.009, right=0.991, hspace=0.255, wspace=0)
ax = fig.add_subplot(221, projection=wcs)
ax2 = fig.add_subplot(222, projection=wcs)
ax3 = fig.add_subplot(223, projection=wcs)
ax4 = fig.add_subplot(224, projection=wcs)
ax.imshow(best_image)
ax.set_title('Attēls')
ax.set_xlabel('Rektascensija [h:m:s]')
ax.set_ylabel('Deklinācija [deg]')
ax2.imshow(image_sharp)
ax2.set_title('Attēls ar uzlabotu asumu')
ax2.set_xlabel('Rektascensija [h:m:s]')
ax2.set_ylabel('Deklinācija [deg]')
ax3.imshow(best_template_median_intensity)
ax3.set_title('Paraugs')
ax3.set_xlabel('Rektascensija [h:m:s]')
ax3.set_ylabel('Deklinācija [deg]')
ax4.imshow(template_sharp)
ax4.set_title('Paraugs ar uzlabotu asumu')
ax4.set_xlabel('Rektascensija [h:m:s]')
ax4.set_ylabel('Deklinācija [deg]')
fig.tight_layout()

# plot Gaussian blur image
fig = plt.figure(figsize=(16, 8), dpi=100)
# fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
ax = fig.add_subplot(3,2,(1,2), projection=wcs)
ax2 = fig.add_subplot(323, projection=wcs)
ax3 = fig.add_subplot(324, projection=wcs)
ax4 = fig.add_subplot(325, projection=wcs)
ax5 = fig.add_subplot(326, projection=wcs)
ax.imshow(best_image)
ax.set_title('Attēls')
ax.set_xlabel('Rektascensija [h:m:s]')
ax.set_ylabel('Deklinācija [deg]')
ax2.imshow(Gaussian_blur_image_1)
ax2.set_title('Gausa filtrs: \n maskas izmērs (3,3), sigma = 0')
ax2.set_xlabel('Rektascensija [h:m:s]')
ax2.set_ylabel('Deklinācija [deg]')
ax3.imshow(Gaussian_blur_image_2)
ax3.set_title('Gausa filtrs: \n maskas izmērs (5,5), sigma = 0')
ax3.set_xlabel('Rektascensija [h:m:s]')
ax3.set_ylabel('Deklinācija [deg]')
ax4.imshow(Gaussian_blur_image_3)
ax4.set_title('Gausa filtrs: \n maskas izmērs (3,3), sigma = 10')
ax4.set_xlabel('Rektascensija [h:m:s]')
ax4.set_ylabel('Deklinācija [deg]')
ax5.imshow(Gaussian_blur_image_4)
ax5.set_title('Gausa filtrs: \n maskas izmērs (5,5), sigma = 10')
ax5.set_xlabel('Rektascensija [h:m:s]')
ax5.set_ylabel('Deklinācija [deg]')
fig.tight_layout()

# plot Gaussian blur template
fig = plt.figure(figsize=(16, 8), dpi=100)
# fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
ax = fig.add_subplot(3,2,(1,2), projection=wcs)
ax2 = fig.add_subplot(323, projection=wcs)
ax3 = fig.add_subplot(324, projection=wcs)
ax4 = fig.add_subplot(325, projection=wcs)
ax5 = fig.add_subplot(326, projection=wcs)
ax.imshow(best_template_median_intensity)
ax.set_title('Paraugs')
ax.set_xlabel('Rektascensija [h:m:s]')
ax.set_ylabel('Deklinācija [deg]')
ax2.imshow(Gaussian_blur_template_1)
ax2.set_title('Gausa filtrs: \n maskas izmērs (3,3), sigma = 0')
ax2.set_xlabel('Rektascensija [h:m:s]')
ax2.set_ylabel('Deklinācija [deg]')
ax3.imshow(Gaussian_blur_template_2)
ax3.set_title('Gausa filtrs: \n maskas izmērs (5,5), sigma = 0')
ax3.set_xlabel('Rektascensija [h:m:s]')
ax3.set_ylabel('Deklinācija [deg]')
ax4.imshow(Gaussian_blur_template_3)
ax4.set_title('Gausa filtrs: \n maskas izmērs (3,3), sigma = 10')
ax4.set_xlabel('Rektascensija [h:m:s]')
ax4.set_ylabel('Deklinācija [deg]')
ax5.imshow(Gaussian_blur_template_4)
ax5.set_title('Gausa filtrs: \n maskas izmērs (5x5), sigma = 10')
ax5.set_xlabel('Rektascensija [h:m:s]')
ax5.set_ylabel('Deklinācija [deg]')
fig.tight_layout()

# plot Median blur image
fig = plt.figure(figsize=(16, 8), dpi=100)
# fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
ax = fig.add_subplot(131, projection=wcs)
ax2 = fig.add_subplot(132, projection=wcs)
ax3 = fig.add_subplot(133, projection=wcs)
ax.imshow(best_image)
ax.set_title('Attēls')
ax.set_xlabel('Rektascensija [h:m:s]')
ax.set_ylabel('Deklinācija [deg]')
ax2.imshow(Median_blur_image_1)
ax2.set_title('Mediānas filtrs: \n maskas izmērs (3x3)')
ax2.set_xlabel('Rektascensija [h:m:s]')
ax2.set_ylabel('Deklinācija [deg]')
ax3.imshow(Median_blur_image_2)
ax3.set_title('Mediānas filtrs: \n maskas izmērs (5x5)')
ax3.set_xlabel('Rektascensija [h:m:s]')
ax3.set_ylabel('Deklinācija [deg]')
fig.tight_layout()

# plot Median blur template
fig = plt.figure(figsize=(16, 8), dpi=100)
# fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
ax = fig.add_subplot(131, projection=wcs)
ax2 = fig.add_subplot(132, projection=wcs)
ax3 = fig.add_subplot(133, projection=wcs)
ax.imshow(best_template_median_intensity)
ax.set_title('Paraugs')
ax.set_xlabel('Rektascensija [h:m:s]')
ax.set_ylabel('Deklinācija [deg]')
ax2.imshow(Median_blur_template_1)
ax2.set_title('Mediānas filtrs: \n maskas izmērs (3x3)')
ax2.set_xlabel('Rektascensija [h:m:s]')
ax2.set_ylabel('Deklinācija [deg]')
ax3.imshow(Median_blur_template_2)
ax3.set_title('Mediānas filtrs: \n maskas izmērs (5x5)')
ax3.set_xlabel('Rektascensija [h:m:s]')
ax3.set_ylabel('Deklinācija [deg]')
fig.tight_layout()
plt.show() #comment this line to not impact performance testing

print('\n')
#prof.print_stats() #uncomment to print Profiler stats in console
prof.disable()  # don't profile the generation of stats
prof.dump_stats('main.stats')

with open('main_output.txt', 'wt') as output:
    stats = Stats('main.stats', stream=output)
    stats.sort_stats('cumulative', 'time')
    stats.print_stats()
    prof.dump_stats("main.prof")
