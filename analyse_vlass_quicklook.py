#!/usr/bin/env python
from __future__ import division, print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

from astropy.io import ascii, fits
from astropy import wcs
from astropy.constants import c
# from astrotools import get_pixelscale_old as get_pixelscale
from astropy.stats import sigma_clip


import astropy.units as u
from astropy import wcs
from astropy.nddata.utils import Cutout2D
from astropy.visualization import ImageNormalize, ZScaleInterval


EXAMPLES = '''


 '''


description = '''
 Read a file containing the path to the VLASS quicklook images called img
 The optical coordinates of the quasars: ra and dec
 and a name.

 The img name must have the VLASS epoch e.g., VLASS1.1 VLASS 1.2 VLASS2.1 or VLASS2.2
 to correct for systematic flux offsets:

https://science.nrao.edu/vlass/data-access/vlass-epoch-1-quick-look-users-guide
 Flux Density Accuracy: The flux density accuracy for the first campaign of the first epoch (VLASS1.1) and subsequent VLASS campaigns (VLASS1.2, VLASS2.1, VLASS2.2) differ, with VLASS1.1 typically worse (most likely due to a problem with the antenna positions that was fixed for 1.2 and subsequent campaigns). For objects in VLASS1.1 with  flux densities below ≈ 1Jy, the peak flux densities are systematically low by ≈ 15%, and the total flux densities by ≈ 10%, with a systematic scatter of ≈ ±8%. Above 1 Jy the flux densities can be very unreliable and should not be used. In VLASS1.2 onwards, the corresponding offsets in flux density are about 8% low for the peak flux densities and 3% low for the total flux densities, with no difference in these accuracies for >1 Jy sources.

 Create a figure of X x X arcsecs with a circle highlighting the optical
 position.

If flux_error = -1 showing that is an upper limit
the reported flux_peak = 3sigma rms


 Check if there is a pixel with pixel / 3sigma * 3 > 3. Highlight that.


                '''



def create_figure():
    '''
    Initialize one page of the pdf log.
    '''
    fig1=plt.figure(figsize=(8.27, 11.69), dpi=100)
    ax1 = fig1.add_axes([0.1, 0.7, 0.8, 0.25])
    plt.setp(ax1, xticks=[], yticks=[])
    ax2 = fig1.add_axes([0.15, 0.25, 0.75, 0.4])

    return fig1, (ax1, ax2)#, ax3)


def draw_beam(ax, bmin, bmaj, bpa, loc='lower left',
    facecolor='white', edgecolor='black',
        hatch='//////', pad=0.5):
    """
    Draw an ellipse of bmin bmaj in data coordinates (pixels)
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse
    ae = AnchoredEllipse(ax.transData, width=bmin, height=bmaj, angle=bpa,
                         loc=loc, pad=pad, borderpad=0.4,
                         frameon=True,)
    ae.ellipse.set(facecolor=facecolor, hatch=hatch,
                edgecolor=edgecolor)
    ax.add_artist(ae)



def get_distance(x1, y1, x2, y2, scale):
    '''
    Get the distance in arcsecs between the radio and optical positions
    It calculates the distance in pixels and then uses the scale to
    transform to arcsecs
    '''
    distance = np.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)

    return distance * scale


def get_rms_from_image(hdu):
    '''
    Calculates the rms from an image as the std after clipping the image for
    outliers using a sigma clipping algorith with sigma=3 because we assume
    a detection with SN=2.5.
    '''
    try:
        image = hdu[0].data * 1e6 #to microJy
    except TypeError:
        image = hdu[1].data * 1e6 #to microJy
    rms0 = np.std(image)
    print("Initial RMS", rms0)
    image_clipped = sigma_clip(image, sigma=2.5, maxiters=5, cenfunc='median',
               axis=None, copy=True)
    rmsc = np.std(image_clipped)
    print("RMS After clipping", rmsc)

    return rmsc




def make_centered_stamp(ax, hdu, ra, dec, size=60, aperture=4,
                         output='stamp.png', title=''):
    '''

    make  a png file of the 20arcsec x 20arcsec stamp and plot
    an specified aperture in arcsecs when ra and dec is not None

    '''

    if (ra is not None) and (dec is not None):
        ra = float(ra)
        dec = float(dec)
    else:
        print("reading the coordinates from the header")
        ra, dec = get_coord_hdr(hdu[0].header)

    try:
        image = hdu[0].data * 1e6 #to microJy
    except TypeError:
        print("hdu[1].data")
        image = hdu[1].data * 1e6 #to microJy

    if len(image.shape) > 2:
        image = image[0][0]
    hdr = hdu[0].header


    #Get the optical position in pixels
    wcs_img = wcs.WCS(hdr)
    wcs_img = wcs_img.celestial

    xc, yc = wcs_img.wcs_world2pix(ra, dec, 0)
    #get pixel scale
    pixel_scale_degrees = np.sqrt((wcs_img.pixel_scale_matrix**2).sum(axis=0, dtype=float)).mean()
    scale = pixel_scale_degrees * 3600 #to arcsec/pix
    # print("scale", scale)
    aperture /= scale
    size=size / scale #pixels

    #centering on the target
    x1 = int(xc - (0.5 * size)); x2 = int(xc + (0.5 * size))
    y1 = int(yc - (0.5 * size)); y2 = int(yc + (0.5 * size))
    # print(x1,x2,y1,y2)
    image = image[y1:y2, x1:x2]
    # print(image.shape)
    try:
        zscale = ZScaleInterval()
        z1, z2 = zscale.get_limits(image)
        simg= ax.imshow(image, origin='lower', cmap=cm.gray,
                               vmin=z1, vmax=z2, interpolation='none')
        # print('z1,z2', z1, z2)
    except Exception as e:
        print(str(e))
        print("TRYING GRAYSCALE")
        simg = ax.imshow(image, origin='lower', cmap=cm.gray)

    ##beam
    bmin = hdr['BMIN'] * 3600 / scale #pixels
    bmax = hdr['BMAJ'] * 3600 / scale #pixels
    bpa = hdr['BPA']
    # draw_beam(ax, bmin, beams['BMAJ'], beams['BPA'])
    draw_beam(ax, bmin, bmax, bpa)


    fig.colorbar(simg, label=r'Flux/beam ($\mu Jy$)')

    circle=plt.Circle((xc-x1, yc-y1),
                             aperture, color='y', fill=False, lw=1.5)
    fig.gca().add_artist(circle)
    ax.set_xlabel('PIXELS')
    ax.set_ylabel('PIXELS')

    ysize, xsize = image.shape
    x1s = int((xsize * 0.5) - aperture)
    x2s = int((xsize * 0.5) + aperture)
    y1s = int((ysize * 0.5) - aperture)
    y2s = int((ysize * 0.5) + aperture)

    #check S/N>3 in sub_img
    sub_img = image[y1s:y2s, x1s:x2s]

    #print(sub_img.shape)
    max_flux = np.max(sub_img)
    maxind = np.unravel_index(sub_img.argmax(), sub_img.shape)


    #TO the original image

    maxindorig = np.int_((maxind[0] +y1s, maxind[1] + x1s))# + (y1, x1)

    ax.plot(maxindorig[1], maxindorig[0], '+', markersize=20)

    dist = get_distance(xc-x1, yc-y1, maxindorig[1], maxindorig[0], scale)
    print("Dist in arcsecs", dist)

    return scale, maxindorig, max_flux, dist



def write_information(ax, i, imgname):
    '''
    Write the information of the image in the upper part of PDF log
    '''



    lines = [#r'redshift: {0:.2f}'.format(row['redshift']),
            'image scale: {0:.2f} arcsec/pix' .format(scale),
           'max pixel: {1:d}, {0:d} ' .format(max_indices[0], max_indices[1]),
           'separation: {0:.1f} arcsec' .format(separation[i])
             ]

    peak_flux = peak[i]
    peak_error = rms_img
    if "VLASS1.1" in imgname:
        print("VLASS 1.1 Correcting flux by 15%")
        peak_flux_corrected = peak_flux*1.15
        peak_error_corrected = peak_error*1.15
    else:
        print("VLASS epoch >1.1 so correcting by 8%")
        peak_flux_corrected = peak_flux*1.08
        peak_error_corrected = peak_error*1.08
    lines2 = [
              r'flux peak image ($\mu Jy$): {0:.1f} pm {1:.1f}'. format(peak_flux, peak_error),
              r'S/N image = {0:.1f} '.format(SNimage[i]),
              r'VLASS corrected flux: ($\mu Jy$): {0:.1f} pm {1:.1f}'. format(peak_flux_corrected, peak_error_corrected)
              ]

    (xt, yt) = (0.08, 0.95)
    for line in lines:
        ax.text(xt, yt, line, transform=ax.transAxes,
              horizontalalignment='left', color='black',
              fontsize='x-large',
              verticalalignment='top')
        yt-=0.1

    #if (SN[i] >= 3.0) | (SNimage[i] >=3.0):
    if (SNimage[i] >=3.0):
         for line in lines2:
            ax.text(xt, yt, line, transform=ax.transAxes,
              horizontalalignment='left', color='black',
              fontsize='x-large', verticalalignment='top',
              fontweight='bold')
            yt-=0.1
    else:
        for line in lines2:
            ax.text(xt, yt, line, transform=ax.transAxes,
              horizontalalignment='left', color='black',
              fontsize='x-large', verticalalignment='top',
              fontweight='light')
            yt-=0.15


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES)

    parser.add_argument('info', type=str,
                      help='Fits or text file containing the information required.\
                      The columns needed are: name (just an identification), img (path to image)\
                       ra and dec in degrees')


    parser.add_argument('-o', '--output', type=str, required=False,
                      default='vlass_nodet_analysis',
                      help='Output name. There will be two outputs.\
                      1) a .pdf file with the image plots and information\
                      2) a .txt file with all the information calculated \
                      as a table (default: %(default)s)')

    parser.add_argument('-a', '--aperture_radius', type=float, required=False,
                          default=3.0,
                          help='Size of the aperture in arcsec to look for signal (default: %(default)s)')

    parser.add_argument('-s', '--size_stamp', type=float, required=False,
                      default=60.0,
                      help='Size of the postage stamp in arcsec (default: %(default)s)')


    return parser.parse_args()

if __name__ == '__main__':

    args=parse_arguments()

    #read info file
    table = ascii.read(args.info)
    #the output file will be the same but with new columns

    #create the arrays where to save the final

    #create the pdf log
    pdflog= args.output + ".pdf"
    pdf=PdfPages(pdflog)

    first_rms_img = []
    SN = []
    SNimage = []
    peak = []
    separation = []

    for i, row in enumerate(table):
        print("=" * 50)
        print(row['name'])
        print(row['img'])
        print("=" * 50)
        fig, axs = create_figure()

        hdu = fits.open(row['img'])
        ra = row['ra']
        dec = row['dec']


        scale, max_indices, max_flux, dist = make_centered_stamp(axs[1], hdu, ra, dec,
         size=args.size_stamp, aperture=args.aperture_radius,  output='stamp.png', title='')

        rms_img = get_rms_from_image(hdu)
        SNimage.append(max_flux / rms_img)

        hdu.close()
        fig.suptitle(row['name']+"/"+row['img'])

        first_rms_img.append(rms_img)
        peak.append(max_flux)
        separation.append(dist)
        write_information(axs[0], i, row['img'])
        pdf.savefig(fig)

        plt.close(fig)

    pdf.close()

    print(pdflog, ' created.')

    table['FIRST_img_rms'] = first_rms_img
    #table['FIRST_SN_cat'] = SN
    table['FIRST_SN_img'] = SNimage
    table['FIRST_forced_peak'] = peak
    table['separation'] = separation

    #print(table)
    tablename= args.output + ".tab"
    try:
        table.write(tablename, format='ascii.csv')
    except TypeError:
        tablename= args.output + ".fits"
        table.write(tablename, format='fits')
    print(tablename, "Created")
