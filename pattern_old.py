from astropy.visualization import ZScaleInterval
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
import argparse
from scipy.signal import find_peaks, peak_widths
import glob, os
import utils as ut
from astropy.io import ascii
import aplpy
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord

from IPython import embed 

import random

import tarfile

import sys
import os.path
sys.path.append('/Users/xie/WORKSPACE/software/qso_sel_toolbox/qso_toolbox/')
import image_tools_temp as it
import catalog_tools_temp as ct
from astropy.nddata.utils import Cutout2D
from astropy import wcs

from astroquery.vizier import Vizier

from matplotlib.backends.backend_pdf import PdfPages

try:
  from urllib2 import urlopen #python2
except ImportError:
  from urllib.request import urlopen #python3


'''
EXAMPLE = 
python pattern.py --mode single --filename J145546.64+361414.78_vlass_3GHz_fov120.fits

python pattern.py --mode multiple

'''



def parse_arguments():
    
    parser = argparse.ArgumentParser(
        description='''
        Convert a fits image into a signal image, based on zscale value cut.
        Made to recognize (symmetric) jet pattern.
        
        '''
    )
    
    parser.add_argument('--plot_mode', type=str, default='normal', choices = ['normal', 'fft'], help='In which mode the plot is made from.')
    parser.add_argument('--filename', type=str, required=False, help='Image fits filename.')
    parser.add_argument('--mode', type=str, required=True, choices = ['single', 'multiple', 'count', 'sample'], help='single: display chosen image analysis, multiple: display ds9 images, count: count epoch and source numbers, sample: generating samples with given size and numbers')
    parser.add_argument('--pdf', type=str, default=False, required=False, help='Save result as pdf file.')
    
    return parser.parse_args()

def check_image_size(file):
    '''
    Check if the input image is 120x120, any size smaller than 100x100 is rejected
    '''
    data=fits.getdata(file)
    return data.shape[1], data.shape[0]

def get_zscale_limits(data):
    '''
    Get zscale limit for input data.
    '''
    zscale = ZScaleInterval(krej=2.0, contrast=0.40)
    z1, z2 = zscale.get_limits(data)
    
    return z1, z2

def code_image_pos(zmax, data):
    '''
    Convert the image of pixel value into (0,1) image - 0 for under the value of zscale_max, 1 for above.
    '''
    if(zmax<0):
        print('zmax value is negative!')
    else:
        data_pos=np.where(data>zmax, 1, 0)
    
    return(data_pos)

def code_image_neg(zmin, data):
    '''
    Solution for detecting UN-space undersampling
    Convert the image of pixel value into (0,1) image - 0 for above the value of zscale_max, 1 for under.
    '''
    data_neg=np.where(data<zmin, 1, 0)
    
    return(data_neg)

def finding_peaks(signal_power):
    '''
    Identify peaks in FFT power plot.
    '''
    peaks, properties = find_peaks(signal_power, height=3, width=1, distance=1, prominence=3) # why 3? -typical neg pixels height is 3
    peaks=np.array(peaks)
    
    # if len(peaks)==0:
    #     # too faint and small source
    #     peaks, properties = find_peaks(signal_power, height=3, width=1, distance=1, prominence=3) # why 3? -typical neg pixels height is 3

    # mask=properties['prominences']>=properties['peak_heights']*1.00
    # peaks=peaks[mask]; properties=new_properties(mask, properties)
    
    if len(peaks)==0:
         # too faint and small source
        peaks, properties = find_peaks(signal_power, height=2, width=1, distance=60, prominence=2) # why 60? -only to select out faint compact object
        
    mask=properties['prominences']>=properties['peak_heights']*1.00-1 # some prominences will be trunctated at the edge
    peaks=peaks[mask]; properties=new_properties(mask, properties)

    if len(peaks)>1:
        peaks, properties=delete_mirror_peaks(peaks, properties)
        
    # only those prominences higher than height*0.50 can be selected

    
    #peaks, _ = find_peaks(signal_power, height=2, width=1, distance=2)
    
    return(peaks, properties)

def new_properties(mask, properties):
    '''
    Create a new properties using boolean masks
    '''

    new_properties={}
    
    for i in properties.keys():
        if isinstance(properties[i], np.ndarray):
            new_properties[i]=properties[i][mask]
    
    return new_properties
        
    
def delete_mirror_peaks(peaks, properties):
    '''
    Get rid of duplicate peaks that are not washed out by prominences because they are of same heights
    '''
    for i in range(len(peaks)):
        for j in range(len(peaks)):
            if i!=j and j>i:
                if properties['peak_heights'][i]==properties['peak_heights'][j]:
                    if properties['prominences'][i]==properties['prominences'][j]:
                        if properties['peak_heights'][i]==properties['prominences'][i]:
                            if properties['left_bases'][i]==properties['left_bases'][j]:
                                if properties['right_bases'][i]==properties['right_bases'][j]:
                                    if properties['widths'][i]==properties['widths'][j]:
                                        offset_i=np.abs(peaks[i]-60); offset_j=np.abs(peaks[j]-60)
                                        if offset_i<=offset_j:
                                            # select only the one that is closer to the center
                                            peaks[j]=-1
                                        else:
                                            peaks[i]=-1
    properties=new_properties((peaks>-1), properties)          
    peaks=np.delete(peaks, np.where(peaks==-1))          
            
    return peaks, properties
                        
def wash_cross(signal_power_x, signal_power_y):
    '''
    Clean cross pattern figures.
    '''
    
    _peaks_x, _ = find_peaks(signal_power_x, height=[1,2])
    _peaks_y, _ = find_peaks(signal_power_y, height=[1,2])

    cross_peaks_x = len(_peaks_x) # number of suspicious cross peaks
    cross_peaks_y = len(_peaks_y)
    cross_peaks=cross_peaks_x+cross_peaks_y
    threshold=10 # at most it can host two cross peaks (height<=2) on two direction (X/Y) added
    
    if cross_peaks>threshold:
        return False
    else:
        return True 

def wash_disturb_morph(signal_power_x, signal_power_y):
    '''
    Clean disturbed central morphology
    only for extended objects
    '''
    
    _peaks_x, _ = find_peaks(signal_power_x, height=2, width=1, distance=2) 
    _peaks_y, _ = find_peaks(signal_power_y, height=2, width=1, distance=2) 
    
    disturb_peaks_x = len(_peaks_x)# number of suspicious peaks related to disturbed morphology
    disturb_peaks_y = len(_peaks_y)# number of suspicious peaks related to disturbed morphology
    peaks_x, _=finding_peaks(signal_power_x); peaks_y, _=finding_peaks(signal_power_y)
    disturb_peaks=disturb_peaks_x+disturb_peaks_y-len(peaks_x)-len(peaks_y)
    threshold=5 # at most it can host 5 cross peaks (height<=2)
    
    if disturb_peaks>threshold:
        return False
    else:
        return True
    
def wash_black_pixels(filename):
    '''
    Clean black pixels via UV-space undersampling
    '''
    data=fits.getdata(filename)
    z1,z2=get_zscale_limits(data)
    
    # if source is too faint (width<=5), then do not use this quality check
    _, _, signal_x, signal_y, _, _=count_signal(filename)
    _, properties_x=find_peaks(signal_x, height=3, width=1, rel_height=0.9)
    _, properties_y=find_peaks(signal_y, height=3, width=1, rel_height=0.9)

    
    data_neg=code_image_neg(z1, data)
    data_pos=code_image_pos(z2, data)
    neg_signal_x=np.sum(data_neg, axis=0)
    #neg_signal_y=np.sum(data_neg, axis=1) # essentially the same as x
    pos_signal_x=np.sum(data_pos, axis=0)
    #pos_signal_y=np.sum(data_pos, axis=1)
    
    sn_ratio=np.sum(neg_signal_x)/np.sum(pos_signal_x)
    
    
    try:
        if ((np.max(properties_x['widths'])<=5)) and (int(np.max(properties_y['widths'])<=5)):
            # too faint source
            return True, sn_ratio 
    except:
        pass
    
    
    if sn_ratio>0.30:
        return False, sn_ratio
    else:
        return True, sn_ratio
    
def download_image(url, savefolder, imagename):
    '''
    Download image and save as the imagename
    '''
    datafile = urlopen(url)
    file=datafile.read()
    all_filename=savefolder+imagename+'.fits'
    output=open(all_filename, 'wb')
    output.write(file)
    output.close()
    
    return all_filename

def download_wise_image(url, savefolder, imagename):
    '''
    Download WISE image and save as the imagename
    Difference: we need to unzip the file
    '''
    datafile = urlopen(url)
    file=datafile.read()
    all_filename=savefolder+imagename+'.tar.gz'
    output=open(all_filename, 'wb')
    output.write(file)
    output.close()
    
    os.mkdir(savefolder+'/'+imagename)
    with tarfile.open(all_filename, "r") as tar:
        #embed()
        img_name=tar.getnames()[0]
        tar.extractall(savefolder+'/'+imagename)
        tar.close()
    
    all_filename=savefolder+'/'+imagename+'/'+imagename+'.fits'
    os.rename(savefolder+'/'+imagename+'/'+img_name, all_filename)
    
    return all_filename   
    
def find_best_optical(filename, ra, dec, fov):
    '''
    Find best optical counterpart image (if any)
    Order: DECaLS, PS1, SkyMapper
    '''
    savefolder='/Users/xie/WORKSPACE/jet_pattern/opticals/'
    band = 'z'
    
    for survey in ['DECaLS DR9 z', 'PS1 DR1 Stack z', 'SkyMapper DR2 z']:
        all_filename=savefolder+filename+'_'+survey+'.fits'
        if(os.path.exists(all_filename)) is True:
            return all_filename, survey
              
    # check and download
    try:
        survey='DECaLS DR9 z'
        url=ct.get_decals_image_url(ra, dec, fov, band, verbosity=0) 
        all_filename=savefolder+filename+'_'+survey+'.fits'
        if(os.path.exists(all_filename)) is False:
            all_filename=download_image(url, savefolder, filename+'_'+survey)
        return all_filename, survey
    except:
        try:
            survey='PS1 DR1 Stack z'
            url=ct.get_ps1_image_cutout_url(ra, dec, fov, band, verbosity=0)
            all_filename=savefolder+filename+'_'+survey+'.fits'
            if(os.path.exists(all_filename)) is False:
                all_filename=download_image(url[0], savefolder, filename+'_'+survey) 
            return all_filename, survey
        except:
            try:
                survey='SkyMapper DR2 z'
                url=ct.get_skymapper_deepest_image_url(ra, dec, fov, band, verbosity=0)
                all_filename=savefolder+filename+'_'+survey+'.fits'
                if(os.path.exists(all_filename)) is False:
                    all_filename=download_image(url, savefolder, filename+'_'+survey)  
                return all_filename, survey     
            except: 
                return None, None
    
def find_best_wise(filename, ra, dec, fov):
    '''
    Find WISE image from unWISE (neoWISE)
    '''
    savefolder='/Users/xie/WORKSPACE/jet_pattern/neowise/'
    band = '1' # W1 is the deepest band
    data_release='neo7'
    npix=int(fov/(0.00076389*3600)) # From wise image header, CD Matrix
    
    all_filename=savefolder+filename+'_WISE_'+band+'/'+filename+'_WISE_'+band+'.fits'
    if(os.path.exists(all_filename)) is True:
        return all_filename
    
            
    url=ct.get_unwise_image_url(ra, dec, npix, band, data_release, filetype="image")
    # all_filename=savefolder+filename+'_WISE_'+band+'_'+'.fits'
    all_filename=download_wise_image(url, savefolder, filename+'_WISE_'+band)
    return all_filename

def find_best_r_band_optical(filename, ra, dec, fov):
    '''
    Find best optical counterpart image (if any)
    Order: DECaLS, PS1, SkyMapper
    For only R band
    '''
    savefolder='/Users/xie/WORKSPACE/jet_pattern/opticals/'
    band = 'r'
    
    for survey in ['DECaLS DR9 r', 'PS1 DR1 Stack r', 'SkyMapper DR2 r']:
        all_filename=savefolder+filename+'_'+survey+'.fits'
        if(os.path.exists(all_filename)) is True:
            return all_filename, survey
              
    # check and download
    try:
        survey='DECaLS DR9 r'
        url=ct.get_decals_image_url(ra, dec, fov, band, verbosity=0) 
        all_filename=savefolder+filename+'_'+survey+'.fits'
        if(os.path.exists(all_filename)) is False:
            all_filename=download_image(url, savefolder, filename+'_'+survey)
        return all_filename, survey
    except:
        try:
            survey='PS1 DR1 Stack r'
            url=ct.get_ps1_image_cutout_url(ra, dec, fov, band, verbosity=0)
            all_filename=savefolder+filename+'_'+survey+'.fits'
            if(os.path.exists(all_filename)) is False:
                all_filename=download_image(url[0], savefolder, filename+'_'+survey) 
            return all_filename, survey
        except:
            try:
                survey='SkyMapper DR2 r'
                url=ct.get_skymapper_deepest_image_url(ra, dec, fov, band, verbosity=0)
                all_filename=savefolder+filename+'_'+survey+'.fits'
                if(os.path.exists(all_filename)) is False:
                    all_filename=download_image(url, savefolder, filename+'_'+survey)  
                return all_filename, survey     
            except: 
                return None, None

def plot_signal(filename, plotmode, repeat_number=1):
    '''
    Plot signal with given plot mode.
    '''
    data=fits.getdata(filename)
    z1,z2=get_zscale_limits(data)
    print(np.std(data))
    print(np.mean(data))
    print(z1, z2)
    print(z1-z2)
    data_code=code_image_pos(z2, data)
    signal_x=np.sum(data_code, axis=0)
    signal_x=np.tile(signal_x,repeat_number)
    signal_y=np.sum(data_code, axis=1)
    signal_y=np.tile(signal_y,repeat_number)
    
    # detecting the UV-space undersampling artifacts AKA black cross
    data_neg=code_image_neg(z1, data)
    neg_signal_x=np.sum(data_neg, axis=0)
    neg_signal_y=np.sum(data_neg, axis=1)
    
    # transfer to Fourier space for x-axis
    fft_signal_x = np.fft.fft(signal_x)
    fft_signal_power_x = np.abs(fft_signal_x)

    # transfer to Fourier space for y-axis
    fft_signal_y = np.fft.fft(signal_y)
    fft_signal_power_y = np.abs(fft_signal_y)
    
    
    freq=np.fft.fftfreq(len(data[1])*repeat_number, d=1)
    
    # if plotmode=='normal':
    #     plt.plot(np.arange(len(data[1])), signal)
    # if plotmode=='fft':
    #     fft_signal = np.fft.fft(signal)
    #     fft_signal_power = np.abs(fft_signal)
    #     plt.plot(np.arange(len(data[1])), fft_signal_power)
    
    xaxis=np.arange(len(data[1])*repeat_number)
    
    fig, axs = plt.subplots(2, 2, figsize=(20,8))
    
    # plot x-axis signal line and peaks
    axs[0, 0].plot(xaxis, signal_x, color='#1A85FF', linewidth=3, zorder=-1)
    peaks_x, _=finding_peaks(signal_x)
    axs[0, 0].scatter(xaxis[peaks_x], signal_x[peaks_x], s=100, marker="x", color='#D41159', linewidths=3)
    
    # data<zmin negative line
    axs[0, 0].plot(xaxis, neg_signal_x, color='black', alpha=1.00)
    print('x-axis black pixels sum:{}'.format(np.sum(neg_signal_x)))
    
    # plot width
    width_x=get_peak_width(peaks_x, signal_x)
    axs[0, 0].hlines(*width_x[1:], color="C2")
    # print(width_x)
    # print(width_x[0])
    
    if(len(peaks_x)>1):
        # multiple peaks, show their distance between each other
        for i, peak in enumerate(peaks_x):  
            if i==0:
                i=i+1  
                continue
            width=peaks_x[i]-peaks_x[i-1]   
            #axs[0, 0].text(peak-0.5*width, 1, int(width))
        
    for i, peak in enumerate(peaks_x):
        #for each peak plot their width
        axs[0, 0].text(peak, width_x[1][i]+0.5, int(width_x[0][i]))
    
    axs[0, 0].set_title('signal in X direction')
    axs[1, 0].plot(np.abs(freq), fft_signal_power_x, color='C1')
    #peaks_x=finding_peaks(fft_signal_power_x)
    #axs[1, 0].plot(np.abs(freq)[peaks_x], fft_signal_power_x[peaks_x], "x", color='C1')

    # plot y-axis signal line and peaks
    axs[0, 1].plot(xaxis, signal_y, color='#1A85FF', linewidth=3, zorder=-1)
    peaks_y, _=finding_peaks(signal_y)
    axs[0, 1].scatter(xaxis[peaks_y], signal_y[peaks_y], s=100, marker="x", color='#D41159', linewidths=3)
    
    # data<zmin negative line
    axs[0, 1].plot(xaxis, neg_signal_y, color='black', alpha=1.00)
    print('y-axis black pixels sum:{}'.format(np.sum(neg_signal_y)))
    
    if(len(peaks_y)>1):
            # multiple peaks, show their distance between each other
        for i, peak in enumerate(peaks_y):  
            if i==0:
                i=i+1  
                continue
            width=peaks_y[i]-peaks_y[i-1]   
            #axs[0, 1].text(peak-0.5*width, 1, int(width))
    
    # plot width
    width_y=get_peak_width(peaks_y, signal_y)
    axs[0, 1].hlines(*width_y[1:], color="C2")
    
    for i, peak in enumerate(peaks_y):
        #for each peak plot their width
        axs[0, 1].text(peak, width_y[1][i]+0.5, int(width_y[0][i]))
    
    axs[0, 1].set_title('signal in Y direction')
    axs[1, 1].plot(np.abs(freq), fft_signal_power_y, color='C9')
    #peaks_y=finding_peaks(fft_signal_power_y)
    #axs[1, 1].plot(np.abs(freq)[peaks_y], fft_signal_power_y[peaks_y], "x", color='C9')
    
    
    fig.suptitle(filename)    
    
    if(if_pdf=='True'):
        plt.savefig('./'+str(filename)+'.pdf', dpi=200)
        plt.clf()
    else:
        plt.show()
        plt.clf()
    
def find_final_class(filenames, flags, quality_flags, dynm_ratios, sn_ratios):
    '''
    Find final class of the source
    '''
    if isinstance(flags, str)==False:
        # multi-epoch source
        if (quality_flags[0]>quality_flags[1]):
            if(quality_flags[1]<2):
                final_flag=flags[1]
            elif(check_dynm_ratios(dynm_ratios)):
                final_flag='COMPACT'
            # else:
            #     final_flag='VISUAL NEEDED'
        elif (quality_flags[0]<quality_flags[1]):
            if(quality_flags[0]<2):
                final_flag=flags[0]
            elif(check_dynm_ratios(dynm_ratios)):
                final_flag='COMPACT'
            # else:
            #     final_flag='VISUAL NEEDED'
        elif (quality_flags[0]==quality_flags[1]):
            if (flags[0]==flags[1] and quality_flags[0]==0):
                final_flag=flags[0]
            else:
                if(flags[0]!=flags[1] and quality_flags[0]==0):
                    _, sn_ratio_1=wash_black_pixels(filenames[0])
                    _, sn_ratio_2=wash_black_pixels(filenames[1])
                    if sn_ratio_1>=sn_ratio_2:
                        #this is noise/signal ratio
                        final_flag=flags[1]
                    else:
                        final_flag=flags[0]
                elif(check_dynm_ratios(dynm_ratios)):
                    final_flag='COMPACT'
                else:
                    _, sn_ratio_1=wash_black_pixels(filenames[0])
                    _, sn_ratio_2=wash_black_pixels(filenames[1])
                    if sn_ratio_1>=sn_ratio_2:
                        #this is noise/signal ratio
                        final_flag=flags[1]
                    else:
                        final_flag=flags[0]
        elif(max(sn_ratios)<5.0):
            final_flag='NON-DETECTION'
                    
    else:
        # single-epoch source
        if(quality_flags<2):
            final_flag=flags
        elif(check_dynm_ratios(dynm_ratios)):
            final_flag='COMPACT'
        elif(sn_ratios<5.0):
            final_flag='NON-DETECTION'
        else:
            final_flag=flags
            
    return final_flag

def check_dynm_ratios(dynm_ratios):
    '''
    Check the dynamical ratios if >10, which suggests it is a compact source
    '''
    if(isinstance(dynm_ratios, np.float32)==False):
        # multi-epoch source
        if min(dynm_ratios)>=10.0:
            # all epochs need to be >=10
            return True
    else:
        # single-epoch source
        if dynm_ratios>=10.0:
            return True
        
    return False
    
def write_image_information(ax, filename, ra, dec, peak_number_x, peak_number_y, flag, quality_flag, dynm_ratios, sn_ratios, final_flag, dists):
    '''
    Write the information of the image in the upper part of PDF log
    '''
    if isinstance(filename, str)==False:
                
        # multi(two)-epoch source
        lines = ['{0:s}' .format(filename[0]),
                'coord= {0:.5f}d, {1:.5f}d | offset= {2:.2f} (VLASS1) - {3:.2f} (VLASS2)'.format(ra, dec, dists[0], dists[1]),
                'dynm ratios= {0:.1f} (VLASS1) - {1:.1f} (VLASS2)'.format(dynm_ratios[0], dynm_ratios[1]),
                'SN ratios= {0:.1f} (VLASS1) - {1:.1f} (VLASS2)'.format(sn_ratios[0], sn_ratios[1]),
                'class= {0:s} ({2:d}) - {1:s} ({3:d})'.format(flag[0], flag[1], quality_flag[0], quality_flag[1]),
                'final class= {0:s}'.format(final_flag)
                ]
        
        #print('{},{},{}'.format(filename[1], flag[0], flag[1]))
        
    else:
        # single-epoch source
        lines = ['{0:s}' .format(filename),
                'coord= {0:.5f}d, {1:.5f}d | offset= {2:.2f}'.format(ra, dec, dists),
                'dynm ratios= {0:.1f}'.format(dynm_ratios),
                'SN ratios= {0:.1f}'.format(sn_ratios),
                'class= {0:s} ({1:d})'.format(flag, quality_flag),
                'final class= {0:s}'.format(final_flag)
                ]        

    (xt, yt) = (0.00, 1.00)
    for line in lines:
        ax.text(xt, yt, line, transform=ax.transAxes,
              horizontalalignment='left', color='black',
              fontsize='x-large',
              verticalalignment='top')
        yt-=0.20
    
    ax.axis('off')
   
def write_image_info_catalog(ax, cat_name, info_dict, phot):
    '''
    Write the catalog info on the pdf report
    '''
    td = info_dict
    lines=''
    if cat_name=='KDB':
        lines = ['KDEBLLACS name: {0:s}'.format(td['name']),
                 'mag_r: {0:.2f}pm{3:.2f} | mag_z: {1:.2f}pm{4:.2f} | mag_W1: {2:.2f}pm{5:.2f}'.format(phot['r_mag'], phot['z_mag'], phot['w1_mag'], phot['e_r_mag'], phot['e_z_mag'], phot['e_w1_mag']),
                'W1-W2={0:.2f} | W2-W3={1:.2f}'.format(td['W1-W2'], td['W2-W3']),
                'Radio counterpart: {0:s}'.format(td['Radio']),
                'Radio flux density: {0:.1f}'.format(td['S']),
                'q12:=log(S12um/S(radio)={0:.2f}'.format(td['q12'])
                ]
        
    if cat_name=='ROMA':
        lines = ['Roma-BzCat name: {0:s}'.format(td['Name']),
                 'mag_r: {0:.2f}pm{3:.2f} | mag_z: {1:.2f}pm{4:.2f} | mag_W1: {2:.2f}pm{5:.2f}'.format(phot['r_mag'], phot['z_mag'], phot['w1_mag'], phot['e_r_mag'], phot['e_z_mag'], phot['e_w1_mag']),
                 'redshift: {0:.3f}{1:s} | R-band magnitude: {2:.1f}'.format(td['z'], td['u_z'], td['Rmag']),
                 'Roma classification: {0:s}'.format(td['Class']),
                 'Radio flux density at 1.4/0.843GHz (mJy): {0:.0f} | at 143GHz (mJy): {1:.1f}'.format(td['FR'],td['F143']),
                 'X-ray 0.1-2.4keV (fW/m2): {0:.2f} | Fermi 1-100GeV (ph/cm2/s): {1:.1E}'.format(td['FX'],td['FF']),
                 'Spectral index radio-optical: {0:.3f}'.format(td['aro'])
                 ]
        
    if cat_name=='WIB2':
        lines = ['WIBRALS name: {0:s}'.format(td['name']),
                 'mag_r: {0:.2f}pm{3:.2f} | mag_z: {1:.2f}pm{4:.2f} | mag_W1: {2:.2f}pm{5:.2f}'.format(phot['r_mag'], phot['z_mag'], phot['w1_mag'], phot['e_r_mag'], phot['e_z_mag'], phot['e_w1_mag']),
                 'W1-W2={0:.2f}pm{3:.2f} | W2-W3={1:.2f}pm{4:.2f} | W3-W4={2:.2f}pm{5:.2f}'.format(td['W1-W2'], td['W2-W3'], td['W3-W4'], td['e_W1-W2'], td['e_W2-W3'], td['e_W3-W4']),
                 'Score on BZB: {0:.2f} | MIX: {1:.2f} | BZQ: {2:.2f}'.format(td['SBZB'], td['SMIX'], td['SBZQ']),
                 'WIBRALS classification: {0:s} class {1:s}'.format(td['Type'], td['Class']),
                 'Radio counterpart: {0:s}'.format(td['Radio']),
                 'q22:=log(S(22um)/S(radio))={0:.2f}'.format(td['q22'])
                 ]
        
    if cat_name=='WIB1':
        lines = ['WIBRALS name: {0:s}'.format(td['name']),
                 'mag_r: {0:.2f}pm{3:.2f} | mag_z: {1:.2f}pm{4:.2f} | mag_W1: {2:.2f}pm{5:.2f}'.format(phot['r_mag'], phot['z_mag'], phot['w1_mag'], phot['e_r_mag'], phot['e_z_mag'], phot['e_w1_mag']),
                 'W1-W2={0:.2f} | W2-W3={1:.2f} | W3-W4={2:.2f}'.format(td['W1-W2'], td['W2-W3'], td['W3-W4']),
                 'Score on BZB: {0:.2f} | MIX: {1:.2f} | BZQ: {2:.2f}'.format(td['SBZB'], td['SMIX'], td['SBZQ']),
                 'WIBRALS classification: {0:s} class {1:s}'.format(td['Type'], td['Class']),
                 'Radio flux density: {0:.1f} | Radio counterpart: {1:s}'.format(td['S'], td['Radio']),
                 'q22:=log(S(22um)/S(radio))={0:.2f}'.format(td['q22'])
                 ]
        
    (xt, yt) = (0.00, 1.00)
    for line in lines:
        ax.text(xt, yt, line, transform=ax.transAxes,
              horizontalalignment='left', color='black',
              fontsize='x-large',
              verticalalignment='top')
        yt-=0.20
        
    ax.axis('off')
    
def get_all_photometry(ra, dec):
    '''
    Get AllWISE W1, PS1 r and z photometry for a given source
    Put those in a dict
    '''
    phot={}
    r_mag, e_r_mag=get_ps1_mag_cat(ra, dec, 'r')
    if(r_mag<0):
        r_mag=0; e_r_mag=0
    phot['r_mag']=r_mag; phot['e_r_mag']=e_r_mag
    
    z_mag, e_z_mag=get_ps1_mag_cat(ra, dec, 'z')
    if(z_mag<0):
        z_mag=0; e_z_mag=0
    phot['z_mag']=z_mag; phot['e_z_mag']=e_z_mag
    
    w1_mag, e_w1_mag=get_allwise_mag_cat(ra, dec, 'w1')
    phot['w1_mag']=w1_mag; phot['e_w1_mag']=e_w1_mag
    
    return phot

def get_ps1_mag_cat(ra, dec, band):
    
    try:
        try:
            # USE DR2
            url_base = 'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.csv?'
            ps1_cat_url = url_base + 'ra={}&dec={}&radius=0.000556'.format(ra, dec)
            
            # ps1_cat_url += '&nDetections.gte=1' # nDetection higher or equal of 1

            cat_df = pd.read_csv(urlopen(ps1_cat_url))
            cat_band_mag = str(band) + 'MeanPSFMag'
            cat_band_mag_err = cat_band_mag + 'Err'

            return float(cat_df[str(cat_band_mag)][0]), float(cat_df[0][str(cat_band_mag_err)][0])
            
        except:
            #USE DR1
            # some sources are missing in DR2!
            url_base = 'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr1/mean.csv?'
            ps1_cat_url = url_base + 'ra={}&dec={}&radius=0.000556'.format(ra, dec)            

            # ps1_cat_url += '&nDetections.gte=1' # nDetection higher or equal of 1

            cat_df = pd.read_csv(urlopen(ps1_cat_url))
            cat_band_mag = str(band) +  'MeanPSFMag'                   
            cat_band_mag_err = cat_band_mag + 'Err'

            return float(cat_df[str(cat_band_mag)][0]), float(cat_df[str(cat_band_mag_err)][0])
    except:
        return 0, 0

def get_allwise_mag_cat(ra, dec, band):
    
    # use vizier query
    
    Vega_to_AB = [2.699, 3.339, 5.174] # mAB - mVega, W1, W2, W3
    
    if band == 'w1':
        band = 'W1'; delta_Vega_to_AB = Vega_to_AB[0]
    if band == 'w2':
        band = 'W2'; delta_Vega_to_AB = Vega_to_AB[1]
    if band == 'w3':
        band = 'W3'; delta_Vega_to_AB = Vega_to_AB[2]
        
    try:
        
        cat_band_mjd = str(band) + 'MJD'
        cat_band_mag = str(band) + 'mag' # here use 3, means 2.0 arcsecs circular
        cat_band_mag_err = 'e_' + cat_band_mag

        v = Vizier(columns=['all'], catalog="II/328") # retrieve all colums
        result = v.query_region(coord.SkyCoord(ra=ra, dec=dec,
                                                    unit=(u.deg, u.deg),
                                                    frame='icrs'),
                                radius=2.0 * u.arcsecond,
                                catalog=["II/328/"]) # allWISE


        return float(result[0][cat_band_mag]) + delta_Vega_to_AB, float(result[0][cat_band_mag_err])

    except:
        return 0, 0
    
def check_which_catalog(ra, dec):
    '''
    1. Check if this source is from: WIBRALS1/2， KDEBLLACS or Roma
    2. Return a dictionary with all info that will be printed on the pdf report
    '''
    cat_name=''; cat_data=None
    kdebllacs_path='/Users/xie/WORKSPACE/jet_pattern/csv/KDEBLLACS.csv'
    kdb_data = ascii.read(kdebllacs_path, format='csv', fast_reader=False)
    kdb_ra=kdb_data['ra']; kdb_dec=kdb_data['dec']
    kdb_idx=if_match_in_cat(ra, dec, kdb_ra, kdb_dec)
    if kdb_idx != 0:
        # matched to a catalog, return a dict contains all info that needs to be printed
        cat_name="KDB"
        cat_data=kdb_data
        return create_target_dict(kdb_idx, cat_name, cat_data)
    
    roma_path='/Users/xie/WORKSPACE/jet_pattern/csv/roma-bzcat-dr5.csv'
    roma_data = ascii.read(roma_path, format='csv', fast_reader=False)
    roma_ra=roma_data['RAJ2000']; roma_dec=roma_data['DEJ2000']
    roma_idx=if_match_in_cat(ra, dec, roma_ra, roma_dec)
    if roma_idx != 0:
        cat_name="ROMA"
        cat_data=roma_data
        return create_target_dict(roma_idx, cat_name, cat_data)
    
    wibrals2_path='/Users/xie/WORKSPACE/jet_pattern/csv/WIBRALS2.csv'
    wib2_data = ascii.read(wibrals2_path, format='csv', fast_reader=False)
    wib2_ra=wib2_data['ra']; wib2_dec=wib2_data['dec']
    wib2_idx=if_match_in_cat(ra, dec, wib2_ra, wib2_dec)
    if wib2_idx != 0:
        cat_name="WIB2"
        cat_data=wib2_data
        return create_target_dict(wib2_idx, cat_name, cat_data)
    
    wibrals1_path='/Users/xie/WORKSPACE/jet_pattern/csv/WIBRALS1.csv'
    wib1_data = ascii.read(wibrals1_path, format='csv', fast_reader=False)
    wib1_ra=wib1_data['ra']; wib1_dec=wib1_data['dec']
    wib1_idx=if_match_in_cat(ra, dec, wib1_ra, wib1_dec)
    if wib1_idx != 0:
        cat_name="WIB1"
        cat_data=wib1_data
        return create_target_dict(wib1_idx, cat_name, cat_data)
    
def if_match_in_cat(ra, dec, cat_ra, cat_dec):
    '''
    Check if the input ra, dec will be matchede with catalog
    '''
    threshold=0.01 # for a target to be matched with the catalog
    
    target = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    cat = SkyCoord(ra=cat_ra*u.deg, dec=cat_dec*u.deg, frame='icrs')
    idx, sep, _ = target.match_to_catalog_sky(cat) 
    
    if sep.to(u.rad).value<threshold:
        return int(idx) # return this idx positional flag to construct a dictionary
    else:
        return 0
    
def create_target_dict(idx, cat_name, cat_data):
    '''
    Return a dictionary with a target's info in a given catalog, that needs to be presented in the pdf
    '''
    td={} # target_dictionary
    if cat_name=='KDB':
        td['name']=cat_data['name'][idx] # target name in KDEBLLACS
        td['W1-W2']=cat_data['W1-W2'][idx] # W1-W2 AllWISE color index
        td['W2-W3']=cat_data['W2-W3'][idx] # W2-W3 AllWISE color index
        td['Radio']=cat_data['Radio'][idx] # radio counterpart name
        td['S']=cat_data['S'][idx] # radio flux density
        td['q12']=cat_data['q12'][idx] # radio-loudness parameter q12 (q12=log(S_12um_/S_radio_)
        
        return 'KDB', td
        
    if cat_name=='ROMA':
        td['Name']=cat_data['Name'][idx] # target name in Roma-BzCat
        td['z']=cat_data['z'][idx] # redshift
        td['u_z']=cat_data['u_z'][idx] # uncertainty flag on redshift
        td['Rmag']=cat_data['Rmag'][idx] # R-band magnitude (AB?)
        td['Class']=cat_data['Class'][idx] # Roma source classification
        td['FR']=cat_data['FR'][idx] # flux density at 1.4/0.843GHz (mJy)
        td['F143']=cat_data['F143'][idx] # flux density at 143GHz (mJy)
        td['FX']=cat_data['FX'][idx] # X-ray flux 0.1-2.4keV (fW/m2)
        td['FF']=cat_data['FF'][idx] # Fermi flux 1-100GeV (ph/cm2/s)
        td['aro']=cat_data['aro'][idx] # spectral index radio-optical

        return 'ROMA', td
    
    if cat_name=='WIB2':
        td['name']=cat_data['name'][idx] # target name in WIBRALS1
        td['W1-W2']=cat_data['W1-W2'][idx] # W1-W2 AllWISE color index
        td['W2-W3']=cat_data['W2-W3'][idx] # W2-W3 AllWISE color index
        td['W3-W4']=cat_data['W3-W4'][idx] # W2-W3 AllWISE color index
        td['e_W1-W2']=cat_data['e_W1-W2'][idx] # uncertainty in W1-W2
        td['e_W2-W3']=cat_data['e_W2-W3'][idx] # uncertainty in W2-W3
        td['e_W3-W4']=cat_data['e_W3-W4'][idx] # uncertainty in W3-W4
        td['SBZB']=cat_data['SBZB'][idx] # score for BZB (BL Lac) region of the locus
        td['SMIX']=cat_data['SMIX'][idx] # score for MIXED region of the locus
        td['SBZQ']=cat_data['SBZQ'][idx] # score for BZQ (FSRQ) region of the locus
        td['Class']=cat_data['Class'][idx] # class (A=most likely WISE blazar-like source)
        td['Type']=cat_data['Type'][idx] # Spectral type: BZB (BL Lac), BZQ (flat spectrum radio quiet), or MIXED
        td['Radio']=cat_data['Radio'][idx] # radio counterpart name
        td['q22']=cat_data['q22'][idx] # log of 22{mu}m to radio flux densities q_22_ = log(S(22um)/S(radio))
        
        return 'WIB2', td
    
    if cat_name=='WIB1':
        td['name']=cat_data['name'][idx] # target name in WIBRALS2
        td['W1-W2']=cat_data['W1-W2'][idx] # W1-W2 AllWISE color index
        td['W2-W3']=cat_data['W2-W3'][idx] # W2-W3 AllWISE color index
        td['W3-W4']=cat_data['W3-W4'][idx] # W2-W3 AllWISE color index
        td['SBZB']=cat_data['SBZB'][idx] # score for BZB (BL Lac) region of the locus
        td['SMIX']=cat_data['SMIX'][idx] # score for MIXED region of the locus
        td['SBZQ']=cat_data['SBZQ'][idx] # score for BZQ (FSRQ) region of the locus
        td['Class']=cat_data['Class'][idx] # class (A=most likely WISE blazar-like source)
        td['Type']=cat_data['Type'][idx] # Spectral type: BZB (BL Lac), BZQ (flat spectrum radio quiet), or MIXED
        td['Radio']=cat_data['Radio'][idx] # radio counterpart name
        td['S']=cat_data['S'][idx] # radio flux density
        td['q22']=cat_data['q22'][idx] # log of 22{mu}m to radio flux densities q_22_ = log(S(22um)/S(radio))      
        
        return 'WIB1', td
    
def plot_analysis(_ax1, _ax2, all_filename):
    '''
    Plot analysis from the algorithm
    '''     
    
    if('EP01' in all_filename):
        epoch='VLASS1'
    elif('EP02' in all_filename):
        epoch='VLASS2'
    else:
        epoch='ONLY EPOCH'
    pathname=folder+all_filename
    data=fits.getdata(pathname)
    z1,z2=get_zscale_limits(data)
    # print(np.std(data))
    # print(np.mean(data))
    # print(z1, z2)
    # print(z1-z2)
    data_code=code_image_pos(z2, data)
    signal_x=np.sum(data_code, axis=0)
    signal_x=np.tile(signal_x,repeat_number)
    signal_y=np.sum(data_code, axis=1)
    signal_y=np.tile(signal_y,repeat_number)
    
    # detecting the UV-space undersampling artifacts AKA black cross
    data_neg=code_image_neg(z1, data)
    neg_signal_x=np.sum(data_neg, axis=0)
    neg_signal_y=np.sum(data_neg, axis=1)
    
    # # transfer to Fourier space for x-axis
    # fft_signal_x = np.fft.fft(signal_x)
    # fft_signal_power_x = np.abs(fft_signal_x)

    # # transfer to Fourier space for y-axis
    # fft_signal_y = np.fft.fft(signal_y)
    # fft_signal_power_y = np.abs(fft_signal_y)
    
    xaxis=np.arange(np.size(data,1)*repeat_number)
    yaxis=np.arange(np.size(data,0)*repeat_number)
    # freq=np.fft.fftfreq(len(data[1])*repeat_number, d=1)
    
    # if plotmode=='normal':
    #     plt.plot(np.arange(len(data[1])), signal)
    # if plotmode=='fft':
    #     fft_signal = np.fft.fft(signal)
    #     fft_signal_power = np.abs(fft_signal)
    #     plt.plot(np.arange(len(data[1])), fft_signal_power)
    
    #fig=plt.figure(figsize=(10,10))
    
    # plot x-axis signal line and peaks
    _ax1.plot(xaxis, signal_x, color='#1A85FF', linewidth=3, zorder=-1)
    peaks_x, _=finding_peaks(signal_x)
    _ax1.scatter(xaxis[peaks_x], signal_x[peaks_x], s=100, marker="x", color='#D41159', linewidths=3)
    
    # data<zmin negative line
    _ax1.plot(xaxis, neg_signal_x, color='black', alpha=1.00)
    #print('x-axis black pixels sum:{}'.format(np.sum(neg_signal_x)))
    
    # plot width
    width_x=get_peak_width(peaks_x, signal_x)
    _ax1.hlines(*width_x[1:], color="C2")
    # print(width_x)
    # print(width_x[0])
    
    if(len(peaks_x)>1):
        # multiple peaks, show their distance between each other
        for i, peak in enumerate(peaks_x):  
            if i==0:
                i=i+1  
                continue
            width=peaks_x[i]-peaks_x[i-1]   
            #axs[0, 0].text(peak-0.5*width, 1, int(width))
        
    for i, peak in enumerate(peaks_x):
        #for each peak plot their width
        _ax1.text(peak, width_x[1][i]+0.5, int(width_x[0][i]))  
        
    _ax1.set_title('signal in X direction ({})'.format(epoch))
    #ax4.plot(np.abs(freq), fft_signal_power_x, color='C1')
    #peaks_x=finding_peaks(fft_signal_power_x)
    #axs[1, 0].plot(np.abs(freq)[peaks_x], fft_signal_power_x[peaks_x], "x", color='C1')

    # plot y-axis signal line and peaks
    _ax2.plot(yaxis, signal_y, color='#1A85FF', linewidth=3, zorder=-1)
    peaks_y, _=finding_peaks(signal_y)
    _ax2.scatter(yaxis[peaks_y], signal_y[peaks_y], s=100, marker="x", color='#D41159', linewidths=3)
    
    # data<zmin negative line
    _ax2.plot(yaxis, neg_signal_y, color='black', alpha=1.00)
    #print('y-axis black pixels sum:{}'.format(np.sum(neg_signal_y)))
    
    if(len(peaks_y)>1):
            # multiple peaks, show their distance between each other
        for i, peak in enumerate(peaks_y):  
            if i==0:
                i=i+1  
                continue
            width=peaks_y[i]-peaks_y[i-1]   
            #axs[0, 1].text(peak-0.5*width, 1, int(width))
    
    # plot width
    width_y=get_peak_width(peaks_y, signal_y)
    _ax2.hlines(*width_y[1:], color="C2")
    
    for i, peak in enumerate(peaks_y):
        #for each peak plot their width
        _ax2.text(peak, width_y[1][i]+0.5, int(width_y[0][i]))
    
    _ax2.set_title('signal in Y direction ({})'.format(epoch))
    #ax5.plot(np.abs(freq), fft_signal_power_y, color='C9')
    #peaks_y=finding_peaks(fft_signal_power_y)
    #axs[1, 1].plot(np.abs(freq)[peaks_y], fft_signal_power_y[peaks_y], "x", color='C9')
        
def plot_signal_pdf(all_filename, filename, ra, dec, peak_number_x, peak_number_y, flag, quality_flag, _diff_flag, dynm_ratios, sn_ratios, final_class, dists):
    '''
    Plot signal with given plot mode.
    edited from plot_signal.
    '''

    if isinstance(filename, str)==False:
        # multi(two)-epoch source
        
        sorter=np.array(all_filename).argsort() # EP01 first, then EP02
        all_filename=np.array(all_filename)[sorter].tolist() 
        filename=np.array(filename)[sorter].tolist()
        peak_number_x=np.array(peak_number_x)[sorter].tolist()
        peak_number_y=np.array(peak_number_y)[sorter].tolist()
        flag=np.array(flag)[sorter].tolist()
        #print('{}, {}'.format(all_filename[1], flag[1]))
        quality_flag=np.array(quality_flag)[sorter].tolist()
        dynm_ratios=np.array(dynm_ratios)[sorter].tolist()
        sn_ratios=np.array(sn_ratios)[sorter].tolist()
        dists=np.array(dists)[sorter].tolist()
        
        
        pdf, fig1, pdf_file_loc=start_pdf_log(filename[0]) # filename[1] is the same as [0]
        
        #gs=GridSpec(4,2, height_ratios=[1.5,4,4,4]) # 3 rows, 2 columns

        # ax1=fig1.add_axes([0.1,0.88,0.8,0.10]) # First row, all column
        # ax2=fig1.add_axes([0.06,0.37,0.39,0.25]) # Second row, first column
        # ax3=fig1.add_axes([0.55,0.37,0.39,0.25]) 
        # ax4=fig1.add_axes([0.06,0.05,0.39,0.25]) 
        # ax5=fig1.add_axes([0.55,0.05,0.39,0.25]) 
        
        ax1=fig1.add_axes([0.1,0.88,0.8,0.10]) # First row, all column
        ax2=fig1.add_axes([0.06,0.25,0.39,0.17]) # Second row, first column
        ax3=fig1.add_axes([0.55,0.25,0.39,0.17]) 
        ax4=fig1.add_axes([0.06,0.05,0.39,0.17]) 
        ax5=fig1.add_axes([0.55,0.05,0.39,0.17]) 
        ax6=fig1.add_axes([0.1,0.76,0.8,0.10])
        
        # fig1, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2)
        
        # header info in the image
        write_image_information(ax1, filename, ra, dec, peak_number_x, peak_number_y, flag, quality_flag, dynm_ratios, sn_ratios, final_class, dists)
        cat_name, info_dict=check_which_catalog(ra, dec)
        phot=get_all_photometry(ra, dec)
        write_image_info_catalog(ax6, cat_name, info_dict, phot)
        
        plot_analysis(ax2, ax3, all_filename[0]) #EP01
        plot_analysis(ax4, ax5, all_filename[1]) #EP02

        opt_flag=1; wise_flag=1; rb_flag=1 # to check if images are correctly downloaded
        
        try:
            optical_all_filename, survey_z=find_best_optical(filename[0], ra, dec, fov=120)
        except:
            opt_flag=0
            
        try:
            wise_all_filename=find_best_wise(filename[0], ra, dec, fov=120)
        except:
            wise_flag=0
            
        try:    
            optical_r_band_all_filename, survey_r=find_best_r_band_optical(filename[0], ra, dec, fov=120)
        except:
            rb_flag=0
            
        #print(optical_all_filename)
        
        # pos1=[0.075, 0.67, 0.25, 0.25*(8.27/16.69)]
        # pos2=[0.375, 0.67, 0.25, 0.25*(8.27/16.69)]
        # pos3=[0.675, 0.67, 0.25, 0.25*(8.27/16.69)]
        
        pos1=[0.075, 0.61, 0.25, 0.25*(8.27/20.69)]
        pos2=[0.375, 0.61, 0.25, 0.25*(8.27/20.69)] # First row, all vlass images
        pos3=[0.075, 0.47, 0.25, 0.25*(8.27/20.69)]
        pos4=[0.375, 0.47, 0.25, 0.25*(8.27/20.69)]
        pos5=[0.675, 0.47, 0.25, 0.25*(8.27/20.69)]
        
        plot_single_fits_beam(fig1, pos=pos1, filename=all_filename[0], ra=ra, dec=dec, fov=120, title='VLASS1')
        plot_single_fits_beam(fig1, pos=pos2, filename=all_filename[1], ra=ra, dec=dec, fov=120, title='VLASS2')
        if optical_all_filename is not None:
            if rb_flag != 0:
                plot_single_fits_optical(fig1, pos=pos3, filename=optical_r_band_all_filename, ra=ra, dec=dec, fov=120, title=str(survey_r), vlass1_data=all_filename[0], vlass2_data=all_filename[1])
            if opt_flag != 0:
                plot_single_fits_optical(fig1, pos=pos4, filename=optical_all_filename, ra=ra, dec=dec, fov=120, title=str(survey_z), vlass1_data=all_filename[0], vlass2_data=all_filename[1])
            if wise_flag != 0:
                plot_single_fits_optical(fig1, pos=pos5, filename=wise_all_filename, ra=ra, dec=dec, fov=120, title=str('neoWISE 7 W1'), vlass1_data=all_filename[0], vlass2_data=all_filename[1])
        #plot_single_fits_beam(fig2, n_row=1, n_col=3, pos=2, filename=all_filename[0], ra=ra, dec=dec, fov=120)
        
    else:
        # single-epoch source
        pdf, fig1, pdf_file_loc=start_pdf_log(filename) # filename[1] is the same as [0]
        #gs=GridSpec(3,2, height_ratios=[1.5,4,4]) # 3 rows, 2 columns

        # ax1=fig1.add_axes([0.1,0.88,0.8,0.10]) # First row, all column
        # ax2=fig1.add_axes([0.06,0.37,0.39,0.25]) # Second row, first column
        # ax3=fig1.add_axes([0.55,0.37,0.39,0.25]) 
        # # ax4=fig1.add_axes([0.06,0.05,0.39,0.25]) 
        # # ax5=fig1.add_axes([0.55,0.05,0.39,0.25]) 

        # ax1=fig1.add_axes([0.1,0.88,0.8,0.10]) # First row, all column
        # ax2=fig1.add_axes([0.06,0.30,0.39,0.20]) # Second row, first column
        # ax3=fig1.add_axes([0.55,0.30,0.39,0.20]) 
        # ax4=fig1.add_axes([0.06,0.05,0.39,0.25]) 
        # ax5=fig1.add_axes([0.55,0.05,0.39,0.25]) 
        ax1=fig1.add_axes([0.1,0.88,0.8,0.10]) # First row, all column
        ax2=fig1.add_axes([0.06,0.25,0.39,0.17]) # Second row, first column
        ax3=fig1.add_axes([0.55,0.25,0.39,0.17]) 
        ax6=fig1.add_axes([0.1,0.76,0.8,0.10])
        
        # header info in the image
        write_image_information(ax1, filename, ra, dec, peak_number_x, peak_number_y, flag, quality_flag, dynm_ratios, sn_ratios, final_class)
        cat_name, info_dict=check_which_catalog(ra, dec)
        phot=get_all_photometry(ra, dec)
        write_image_info_catalog(ax6, cat_name, info_dict, phot)
        
        plot_analysis(ax2, ax3, all_filename) #the only epoch
        
        opt_flag=1; wise_flag=1; rb_flag=1 # to check if images are correctly downloaded
        
        try:
            optical_all_filename, survey_z=find_best_optical(filename, ra, dec, fov=120)
        except:
            opt_flag=0
            
        try:
            wise_all_filename=find_best_wise(filename, ra, dec, fov=120)
        except:
            wise_flag=0
     
        try:
            optical_r_band_all_filename, survey_r=find_best_r_band_optical(filename, ra, dec, fov=120)
        except:
            rb_flag=0
            
        #print(optical_all_filename)
        
        # pos1=[0.075, 0.67, 0.25, 0.25*(8.27/16.69)]
        # pos2=[0.375, 0.67, 0.25, 0.25*(8.27/16.69)]
        # pos3=[0.675, 0.67, 0.25, 0.25*(8.27/16.69)]

        pos1=[0.075, 0.61, 0.25, 0.25*(8.27/20.69)]
        pos2=[0.375, 0.61, 0.25, 0.25*(8.27/20.69)] # First row, all vlass images
        pos3=[0.075, 0.47, 0.25, 0.25*(8.27/20.69)]
        pos4=[0.375, 0.47, 0.25, 0.25*(8.27/20.69)]
        pos5=[0.675, 0.47, 0.25, 0.25*(8.27/20.69)]

        plot_single_fits_beam(fig1, pos=pos1, filename=all_filename, ra=ra, dec=dec, fov=120, title='VLASS')
        # plot_single_fits_beam(fig1, pos=pos1, filename=all_filename, ra=ra, dec=dec, fov=120, title='VLASS2')
        if optical_all_filename is not None:
            if rb_flag != 0:
                plot_single_fits_optical(fig1, pos=pos3, filename=optical_r_band_all_filename, ra=ra, dec=dec, fov=120, title=str(survey_r), vlass1_data=None, vlass2_data=all_filename)
            if opt_flag != 0:
                plot_single_fits_optical(fig1, pos=pos4, filename=optical_all_filename, ra=ra, dec=dec, fov=120, title=str(survey_z), vlass1_data=None, vlass2_data=all_filename)
            if wise_flag != 0:
                plot_single_fits_optical(fig1, pos=pos5, filename=wise_all_filename, ra=ra, dec=dec, fov=120, title=str('neoWISE 7 W1'), vlass1_data=None, vlass2_data=all_filename)
        #plot_single_fits_beam(fig2, n_row=1, n_col=3, pos=2, filename=all_filename[0], ra=ra, dec=dec, fov=120)
    #fig.suptitle(filename)    
    
    close_pdf_log(pdf, fig1)
    
    plt.close('all')

    return pdf_file_loc
    
    # if(if_pdf=='True'):
    #     plt.savefig('./'+str(filename)+'.pdf', dpi=200)
    # else:
    #     plt.show()

   
def count_signal(filename):
    '''
    Count the peaks in the signal of X/Y.
    '''
    data=fits.getdata(filename)
    z1,z2=get_zscale_limits(data)
    data_code=code_image_pos(z2, data)
    signal_x=np.sum(data_code, axis=0)
    signal_x=np.tile(signal_x,repeat_number) # for generating repeating pattern
    signal_y=np.sum(data_code, axis=1)
    signal_y=np.tile(signal_y,repeat_number) # for generating repeating pattern
    
    peaks_x, properties_x=finding_peaks(signal_x)
    peaks_y, properties_y=finding_peaks(signal_y)
    
    return(peaks_x, peaks_y, signal_x, signal_y, properties_x, properties_y)

def get_peak_width(peaks, signal):
    '''
    evaluate the peak width based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html#scipy.signal.peak_widths
    '''
    width_result = peak_widths(signal, peaks, rel_height=0.9)
    return width_result
    
def read_from_table(table, path):
    '''
    read all the images in the table, and then count the peaks.
    '''
    
    meta_peaks_x=[]; meta_peaks_x=np.array(meta_peaks_x)
    meta_peaks_y=[]; meta_peaks_y=np.array(meta_peaks_y)
    filenames=[]; filenames=np.array(filenames)
    
    os.chdir(path)
    counter=0
    tot_counter=0
    
    ra_name='RAJ2000'; dec_name='DEJ2000'
    catalog = SkyCoord(ra=table[ra_name]*u.degree, dec=table[dec_name]*u.degree) # for matching the entry of table
    
    for file in glob.glob("*vlass*fits"):
        ra = file[1:3]+'h'+file[3:5]+'m'+file[5:10]+'s'
        dec = file[10:13]+'d'+file[13:15]+'m'+file[15:20]+'s'
        if(match_to_images(ra, dec, catalog)):
            tot_counter=tot_counter+1
            peaks_x, peaks_y, signal_x, signal_y, properties_x, properties_y = count_signal(file)
            np.append(meta_peaks_x, peaks_x)
            np.append(meta_peaks_y, peaks_y)            
            flag=' '
            peaks_width_x=get_peak_width(peaks_x, signal_x); peaks_width_y=get_peak_width(peaks_y, signal_y)
            
            #embed()

            if(len(peaks_x)>0 and len(peaks_x)<3):
                if (max(peaks_width_x[0])>10):
                    flag='EXTENDED'; counter=counter+1
                    print(file, len(peaks_x), len(peaks_y), flag)
                    #os.system("open " + file)        
            elif(len(peaks_y)>0 and len(peaks_y)<3):
                if (max(peaks_width_y[0])>10):
                    flag='EXTENDED'; counter=counter+1
                    print(file, len(peaks_x), len(peaks_y), flag)
                    #os.system("open " + file)    
            elif (len(peaks_x)>=3 or len(peaks_y)>=3):           
                if (wash_cross(signal_x)==True and wash_cross(signal_y)==True):                
                    flag='MULTI-PEAKS'; counter=counter+1
                    print(file, len(peaks_x), len(peaks_y), flag)
                    #os.system("open " + file)
        # else:
        #     embed()
    
    print('there are in total {} candidates found.'.format(counter))
    
def count_from_folder(path):
    '''
    read all the images in the folder, and then count the source number - with one or two epochs present in VLASS Quick Look Images.
    '''
    
    number_epochs=0; number_sources=0
    obj_names=[]
    
    #new lists with non-repeat sources
    ras=[]; decs=[]
    filenames=[]


    os.chdir(path)
    
    for file in glob.glob("*vlass*fits"):
        
        #data that needs to be put in the output table
        obj_name=file[:20]
    
        number_epochs+=1
        
        if obj_name in obj_names:
            # only sources with two epochs have EP in filename, thus multi-epoch sources
            
            continue
        
        obj_names.append(obj_name)
        number_sources+=1
        
    # with some equations work, number of single-epoch and double-epoch sources can be calculated as below    
    number_single_sources=2*number_sources-number_epochs
    number_double_sources=number_epochs-number_sources
    
    # d={'filename': filenames, 'ra': ras, 'dec': decs, 'pathname': all_filenames}
    # df=pd.DataFrame(data=d)
    # file_loc='/Users/xie/WORKSPACE/jet_pattern/output/test_output.csv'
    # df.to_csv(file_loc, index=False)
        
    print("[INFO]NUMBER OF EPOCHS:{}, NUMBER OF SOURCES:{}\n[INFO]SINGLE-EPOCH SOURCES NUMBER:{}, TWO-EPOCH SOURCES NUMBER:{}".format(number_epochs, number_sources, number_single_sources, number_double_sources))

def get_pix_center(filename, ra, dec, fov):
    '''
    Get the center position in the pixel coordinate
    They are not always in the center, since there are fits images that are not square
    The positions are determined by the WCS info stored in image's header
    '''
    filename=folder+filename
    data, hdr, _ = it.open_image(filename, ra, dec, fov,
                                   image_folder_path=None,
                           verbosity=0)

    wcs_img = wcs.WCS(hdr)

    pixcrd = wcs_img.wcs_world2pix(ra, dec, 0)
    positions = (np.float(pixcrd[0]), np.float(pixcrd[1]))
    
    img_stamp = Cutout2D(data, positions, size=fov * u.arcsec,
                            wcs=wcs_img)
    cutout_positions = img_stamp.to_cutout_position((np.float(pixcrd[0]), np.float(pixcrd[1])))
    #print("VLASS cutout positions: {}".format(cutout_positions))
    
    return cutout_positions
    
    
def calc_dynm_ratio(filename, ra, dec):
    '''
    Calculate dynamic ratio in a fits image
    https://science.nrao.edu/vlass/data-access/vlass-epoch-1-quick-look-users-guide
    '''
    data=fits.getdata(filename)
    fov_x=np.size(data,0); fov_y=np.size(data,1)
    x, y=np.ogrid[:fov_x,:fov_y]
    #cx=fov_x/2; cy=fov_y/2
    pix_center_pos=get_pix_center(filename, ra, dec, fov=120)
    cx=pix_center_pos[0]; cy=pix_center_pos[1]

    cr=2 # radius to measure peak flux for the center
    r=8 # radius to measure circle flux
    
    mask_center=(x-cx)**2+(y-cy)**2<cr**2
    mask_annulus=((x-cx)**2+(y-cy)**2>(r-1)**2) & ((x-cx)**2+(y-cy)**2<(r+1)**2)
    
    dynm_ratio=np.max(data[mask_center])/np.max(data[mask_annulus])
    
    return dynm_ratio
    
def calc_sn_ratio(filename, ra, dec):
    '''
    from Eduardo's code
    Calculates the rms from an image as the std after clipping the image for
    outliers using a sigma clipping algorith with sigma=3 because we assume
    a detection with SN=2.5.
    '''
    image=fits.getdata(filename)

    fov_x=np.size(image,1); fov_y=np.size(image,0)
    x, y=np.ogrid[:fov_x,:fov_y]
    #cx=fov_x/2; cy=fov_y/2
    pix_center_pos=get_pix_center(filename, ra, dec, fov=120)
    cx=pix_center_pos[0]; cy=pix_center_pos[1]
    
    cr=10 # radius to measure peak flux for the center
    
    mask_center=(x-cx)**2+(y-cy)**2<cr**2
    peak_flux=np.max(image.T[mask_center])
    #print("Peak flux in aperture 10", peak_flux)
    
    
    #print("Initial RMS", rms0)
    image_clipped = sigma_clip(image, sigma=2.5, maxiters=5, cenfunc='median',
               axis=None, copy=True)
    rmsc = np.std(image_clipped)
    #print("RMS After clipping", rmsc)
    sn_ratio=peak_flux/rmsc
    #print("S/N ratio", sn_ratio)
    
    

    return sn_ratio
    
def calc_dist_center(filename, ra, dec, peaks_x, peaks_y):
    '''
    Calculate the position offset of the source
    '''
    cent_x=peaks_x[0]; cent_y=peaks_y[0]
    pix_center_pos=get_pix_center(filename, ra, dec, fov=120)
    cx=pix_center_pos[0]; cy=pix_center_pos[1]
    dist = (cent_x-cx)**2+(cent_y-cy)**2
    
    return dist**0.5
    
def read_from_folder(path):
    '''
    Read all the images in the folder, and then count the peaks.
    '''
    
    # counters
    os.chdir(path)
    counter=0
    tot_counter=0
    ext_two_sd_counter=0
    ext_one_sd_counter=0
    mul_two_sd_counter=0
    mul_one_sd_counter=0
    comp_counter=0
    oth_counter=0
    
    #data that needs to be put in the output table
    ras=[]; decs=[]
    peak_numbers_x=[]; peak_numbers_y=[]
    classes=[]; quality_flags=[]
    filenames=[]
    all_filenames=[]
    dynm_ratios=[]
    sn_ratios=[]
    dists=[]
    # complete_filenames=[] # filename but with all sources, no matter selected out or not
    # complete_all_filenames=[] 
    # complete_peak_numbers_x=[]
    # complete_peak_numbers_y=[]
    
    existed_pdf_file=[]
    
    for pdf_file in glob.glob("*.pdf"):
        existed_pdf_file.append(pdf_file)
          
    for file in glob.glob("*vlass*fits"):      
        #data that needs to be put in the output table
        all_filename=file
        filename=file[0:20]
        ra = file[1:3]+'h'+file[3:5]+'m'+file[5:10]+'s'
        dec = file[10:13]+'d'+file[13:15]+'m'+file[15:20]+'s'
        #change to decimal units
        c = SkyCoord(ra, dec, frame='icrs')
        ra=c.ra.degree; dec=c.dec.degree
        dist=0 # default value
        
        try:

            dynm_ratio=calc_dynm_ratio(file, ra, dec)
            sn_ratio=calc_sn_ratio(file, ra, dec)
            
            tot_counter=tot_counter+1
            # try:
            peaks_x, peaks_y, signal_x, signal_y, properties_x, properties_y = count_signal(file)
            
            # except:
            #     #only known problem: J032040.80-010127.84_vlass_3GHz_fov120_EP01.fits is empty
            #     if(filename=='J032040.80-010127.84'):
            #         pass
            #     else:
            #         print('[WARNING]Porblem encontered at {}'.format(filename))
            #         embed()          
            class_img=' '
            quality_flag=0
            peaks_width_x=get_peak_width(peaks_x, signal_x); peaks_width_y=get_peak_width(peaks_y, signal_y) 
            # print('Peaks:{},{}'.format(peaks_x, peaks_y))
            # CLASSIFICATION
            
            
            if(if_duplicate(ra, dec, ras, decs)==False):                      
                #EXTENDED                           
                if (len(peaks_x)==1 and len(peaks_y)==1):
                    dist=calc_dist_center(file, ra, dec, peaks_x, peaks_y)
                    if(max(signal_x)>5 and max(signal_y)>5) and (max(peaks_width_x[0])>=20 or max(peaks_width_y[0])>=20): # this value represents the width at the peak position
                        if(if_peak_centered(signal_x, signal_y, peaks_x, peaks_y, properties_x, properties_y))==True: # centering peak is less than 3 pixels away
                            class_img='2-SIDE EXTENDED'; counter=counter+1; ext_two_sd_counter=ext_two_sd_counter+1
                            quality_flag=get_quality_flag(file, signal_x, signal_y, extended=True)
                            print(file, len(peaks_x), len(peaks_y), class_img)   
                        else:
                            class_img='1-SIDE EXTENDED'; counter=counter+1; ext_one_sd_counter=ext_one_sd_counter+1
                            quality_flag=get_quality_flag(file, signal_x, signal_y, extended=True)
                            print(file, len(peaks_x), len(peaks_y), class_img)            
                
                    elif(if_offset(peaks_x, peaks_y, properties_x, properties_y, file, ra, dec)):
                        class_img='COMPACT_offset'; counter=counter+1; comp_counter=comp_counter+1
                        quality_flag=get_quality_flag(file, signal_x, signal_y, extended=True)
                        print(file, len(peaks_x), len(peaks_y), class_img)     
                    
                    #COMPACT
                    else:
                        class_img='COMPACT'; counter=counter+1; comp_counter=comp_counter+1
                        quality_flag=get_quality_flag(file, signal_x, signal_y, extended=True)
                        print(file, len(peaks_x), len(peaks_y), class_img)                                 

                # TWO/ONE-SIDED SEPARATED
                elif (len(peaks_x)>=2 or len(peaks_y)>=2 and len(peaks_x)!=0 and len(peaks_y)!=0): 
                    if(if_one_side(signal_x, signal_y)==False):         
                        class_img='2-SIDE SEPARATED'; counter=counter+1; mul_two_sd_counter=mul_two_sd_counter+1
                        quality_flag=get_quality_flag(file, signal_x, signal_y, extended=False)
                        print(file, len(peaks_x), len(peaks_y), class_img)
                    else:
                        class_img='1-SIDE SEPARATED'; counter=counter+1; mul_one_sd_counter=mul_one_sd_counter+1
                        quality_flag=get_quality_flag(file, signal_x, signal_y, extended=False)
                        print(file, len(peaks_x), len(peaks_y), class_img)
                        
                # OTHER                            
                else:
                    class_img='OTHER'; counter=counter+1; oth_counter=oth_counter+1
                    quality_flag=get_quality_flag(file, signal_x, signal_y, extended=False)
                    print(file, len(peaks_x), len(peaks_y), class_img)
                
                if (sn_ratio<2.0):
                    class_img='NON-DETECTION'; counter=counter+1; comp_counter=comp_counter+1
                    quality_flag=get_quality_flag(file, signal_x, signal_y, extended=True)
                    print(file, len(peaks_x), len(peaks_y), class_img)      
                
        except:
            print('[WARNING]failed for {}, skipping...'.format(file))
                                
        #now append everything - for one single source                        
        ras.append(ra) ; decs.append(dec)
        peak_numbers_x.append(len(peaks_x))
        peak_numbers_y.append(len(peaks_y))
        quality_flags.append(quality_flag); classes.append(class_img)
        filenames.append(filename)
        all_filenames.append(all_filename)
        dynm_ratios.append(dynm_ratio)
        sn_ratios.append(sn_ratio)
        dists.append(dist)
        
        
        # complete_filenames.append(filename)
        # complete_all_filenames.append(all_filename)
        # complete_peak_numbers_x.append(len(peaks_x))
        # complete_peak_numbers_y.append(len(peaks_y))
            

    _ras=np.array(ras); _decs=np.array(decs); _peak_numbers_x=np.array(peak_numbers_x); \
        _peak_numbers_y=np.array(peak_numbers_y); _classes=np.array(classes); _quality_flags=np.array(quality_flags);\
            _filenames=np.array(filenames); _all_filenames=np.array(all_filenames); _dynm_ratios=np.array(dynm_ratios); _sn_ratios=np.array(sn_ratios); _dists=np.array(dists)
    #_complete_filenames=np.array(complete_filenames); _complete_all_filenames=np.array(complete_all_filenames); _complete_peak_numbers_x=np.array(complete_peak_numbers_x); _complete_peak_numbers_y=np.array(complete_peak_numbers_y)

    #new lists with non-repeat sources
    ras=[]; decs=[]
    peak_numbers_x=[]; peak_numbers_y=[]
    classes=[]
    filenames=[]
    quality_flags=[]
    diff_flags=[] # check if two epochs of VLASS images give different result
    
    # columns to write in the csv
    output_filename=[]
    output_ra=[]; output_dec=[]
    output_class_v1=[]; output_class_v2=[]
    output_qf_v1=[]; output_qf_v2=[]
    output_diff_flag=[]
    output_pdf_loc=[]
    output_dynm_ratio_v1=[]; output_dynm_ratio_v2=[]
    output_sn_ratio_v1=[]; output_sn_ratio_v2=[]
    output_dist_v1=[]; output_dist_v2=[]
    output_final_class=[]
    
    
    #merge source with two epochs into one
    for k, filename in enumerate(_filenames):
        
        if(filename in filenames):
            #the same source in a different epoch is already registered
            continue
        
        _diff_flag=0
        pos=np.where(_filenames==str(filename))
        if(len(pos[0])>1):
            
            #two-epoch sources
            temp_classes=_classes[pos]
            _diff_flag=''
            for i in range(len(temp_classes)):
                for j in range(len(temp_classes)):
                    if(i!=j):
                        if(temp_classes[i]!=temp_classes[j]):
                            _diff_flag='-1' # different flags in two epochs
                            break
                if(_diff_flag!=0):
                    break
            
            ras.append(_ras[k]) ; decs.append(_decs[k])
            peak_numbers_x.append(_peak_numbers_x[k])
            peak_numbers_y.append(_peak_numbers_y[k])
            classes.append(_classes[k])
            filenames.append(_filenames[k])
            quality_flags.append(_quality_flags[k])
            diff_flags.append(_diff_flag)
            
            # make pdf report, need to send parameters in list
            temp_all_filenames = _all_filenames[pos[0]]
            temp_filenames = _filenames[pos[0]] # save all filenames into a list
            temp_classes = _classes[pos[0]]
            temp_peak_numbers_x = _peak_numbers_x[pos[0]]
            temp_peak_numbers_y = _peak_numbers_y[pos[0]]
            temp_quality_flags = _quality_flags[pos[0]]
            temp_dynm_ratios= _dynm_ratios[pos[0]]
            temp_sn_ratios=_sn_ratios[pos[0]]
            temp_dists=_dists[pos[0]]
            
                
            # pdf_name = temp_filenames[0]+'.pdf'
            # temp_final_class = find_final_class(temp_all_filenames, temp_classes, temp_quality_flags, temp_dynm_ratios, temp_sn_ratios)
            # if pdf_name in existed_pdf_file:
            #     print('[INFO] PDF {} already downloaded!'.format(pdf_name))
            # else:
            #     print("[INFO] Generating PDF card for {}".format(temp_filenames[0]))
            #     pdf_file_loc=plot_signal_pdf(temp_all_filenames, temp_filenames, _ras[k], _decs[k], temp_peak_numbers_x, temp_peak_numbers_y, temp_classes, temp_quality_flags, _diff_flag, temp_dynm_ratios, temp_sn_ratios, temp_final_class, temp_dists)
            #temp_final_class = find_final_class(temp_all_filenames, temp_classes, temp_quality_flags, temp_dynm_ratios, temp_sn_ratios)
            
            try:
                pdf_name = temp_filenames[0]+'.pdf'
                temp_final_class = find_final_class(temp_all_filenames, temp_classes, temp_quality_flags, temp_dynm_ratios, temp_sn_ratios)
                if pdf_name in existed_pdf_file:
                    print('[INFO] PDF {} already downloaded!'.format(pdf_name))
                else:
                    print("[INFO] Generating PDF card for {}".format(temp_filenames[0]))
                    pdf_file_loc=plot_signal_pdf(temp_all_filenames, temp_filenames, _ras[k], _decs[k], temp_peak_numbers_x, temp_peak_numbers_y, temp_classes, temp_quality_flags, _diff_flag, temp_dynm_ratios, temp_sn_ratios, temp_final_class, temp_dists)
                #temp_final_class = find_final_class(temp_all_filenames, temp_classes, temp_quality_flags, temp_dynm_ratios, temp_sn_ratios)
                
            except:
                print('[WARNING] Failed for {}, skipping...'.format(filename))
                #embed()
                pass
            
            # write output csv
            sorter=np.array(temp_all_filenames).argsort() # EP01 first, then EP02
            output_filename.append(_filenames[k])
            output_ra.append(_ras[k]); output_dec.append(_decs[k])
            output_class_v1.append(temp_classes[sorter][0]); output_class_v2.append(temp_classes[sorter][1])
            output_qf_v1.append(temp_quality_flags[sorter][0]); output_qf_v2.append(temp_quality_flags[sorter][1])
            output_diff_flag.append(_diff_flag)       
            #output_pdf_loc.append(pdf_file_loc)  
            output_dynm_ratio_v1.append(temp_dynm_ratios[sorter][0]); output_dynm_ratio_v2.append(temp_dynm_ratios[sorter][1])   
            output_sn_ratio_v1.append(temp_sn_ratios[sorter][0]); output_sn_ratio_v2.append(temp_sn_ratios[sorter][1])
            output_dist_v1.append(temp_dists[sorter][0]); output_dist_v2.append(temp_dists[sorter][1])
            output_final_class.append(temp_final_class)
        
        else:      
            #single-epoch sources
            _diff_flag='1'
            ras.append(_ras[k]) ; decs.append(_decs[k])
            peak_numbers_x.append(_peak_numbers_x[k])
            peak_numbers_y.append(_peak_numbers_y[k])
            classes.append(_classes[k])
            filenames.append(_filenames[k])
            quality_flags.append(_quality_flags[k])
            diff_flags.append(_diff_flag)
            
            _final_class=find_final_class(_all_filenames[k], _classes[k], _quality_flags[k], _dynm_ratios[k], _sn_ratios[k])
            
            # make pdf report
            try:
                pdf_name=_filenames[k]+'.pdf'
                if  pdf_name in existed_pdf_file:
                    print('[INFO] PDF {} already downloaded!'.format(pdf_name))
                else:
                    print("[INFO] Generating PDF card for {}".format(_filenames[k]))
                    pdf_file_loc=plot_signal_pdf(_all_filenames[k], _filenames[k], _ras[k], _decs[k], _peak_numbers_x[k], _peak_numbers_y[k], _classes[k], _quality_flags[k], _diff_flag, _dynm_ratios[k], _sn_ratios[k], _final_class, _dists[k])
            except:
                print('[WARNING]failed for {}, skipping...'.format(filename))
                #embed()
                pass
            
            # pdf_name=_filenames[k]+'.pdf'
            # if  pdf_name in existed_pdf_file:
            #     print('[INFO] PDF {} already downloaded!'.format(pdf_name))
            # else:
            #     print("[INFO] Generating PDF card for {}".format(_filenames[k]))
            #     pdf_file_loc=plot_signal_pdf(_all_filenames[k], _filenames[k], _ras[k], _decs[k], _peak_numbers_x[k], _peak_numbers_y[k], _classes[k], _quality_flags[k], _diff_flag, _dynm_ratios[k], _sn_ratios[k], _final_class)
                
            # write output csv
            output_filename.append(_filenames[k])
            output_ra.append(_ras[k]); output_dec.append(_decs[k])
            output_class_v1.append(_classes[k]); output_class_v2.append('')
            output_qf_v1.append(_quality_flags[k]); output_qf_v2.append('')
            output_diff_flag.append(_diff_flag)  
            #output_pdf_loc.append(pdf_file_loc)   
            output_dynm_ratio_v1.append(_dynm_ratios[k]); output_dynm_ratio_v2.append(-1)  
            output_sn_ratio_v1.append(_sn_ratios[k]); output_sn_ratio_v2.append(-1)
            output_final_class.append(_final_class)
            output_dist_v1.append(_dists[k]); output_dist_v2.append(-1)
            
    d={'filename': output_filename, 'ra': output_ra, 'dec': output_dec, 'class_V1': output_class_v1, 'class_V2': output_class_v2, 'QF_V1': output_qf_v1, 'QF_V2': output_qf_v2, \
        'D_ratio_V1': output_dynm_ratio_v1, 'D_ratio_V2': output_dynm_ratio_v2, 'SN_ratio_V1': output_sn_ratio_v1, 'SN_ratio_V2': output_sn_ratio_v2, 'final_class': output_final_class}
    df=pd.DataFrame(data=d)
    file_loc=output_file
    df.to_csv(file_loc, index=False)
    
    print('[INFO]there are in total {} candidates found for ALL epochs.'.format(counter))
    print('[INFO]{} are 2-SIDE EXTENDED; {} are 1-SIDE EXTENDED;'.format(ext_two_sd_counter, ext_one_sd_counter))
    print('[INFO]{} are 2-SIDE SEPARATED; {} are 1-SIDE SEPARATED;'. format(mul_two_sd_counter, mul_one_sd_counter))
    print('[INFO]{} are COMPACT; {} are OTHER;'. format(comp_counter, oth_counter))
    print('[INFO]all candidates info written into {}'.format(file_loc))
    
def start_pdf_log(filename):
    '''
    Log all plots into a pdf file
    '''
    pdflog=str(filename) + ".pdf"
    pdf=PdfPages(pdflog)

    # fig1, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=4, ncols=2, figsize=(10,20))
    
    fig1=plt.figure(figsize=(8.27, 20.69), dpi=100) # first page
    # fig1=plt.figure(figsize=(10, 18), dpi=100)
    # ax1 = fig1.add_axes([0.1, 0.8, 0.8, 0.15]) # first page, the text lines
    # #plt.setp(ax1, xticks=[], yticks=[])
    # ax2 = fig1.add_axes([0.1, 0.45, 0.75, 0.3])
    # ax3 = fig1.add_axes([0.15, 0.1, 0.75, 0.3])

    # fig2=plt.figure(figsize=(10, 10), dpi=100) # second page

    # title = 'APERTURE PHOTOMETRY ANALYSIS FOR IMAGE: {0:s}'.format(img)
    # for fig in [fig1, fig2]:
    #     fig.suptitle(title)

    return pdf, fig1, pdflog

def close_pdf_log(pdf, fig1):
    '''
    Close of pdf file log
    '''
    pdf.savefig(fig1)
    #pdf.savefig(fig2)
    pdf.close()
    print("[INFO]PDFlog saved")
    
def if_duplicate(ra, dec, ra_all, dec_all):
    '''
    Check if there is already a source within 2'', and not registered in catalog if so 
    '''
    if(len(ra_all)==0): #first element
        return False
    obj=SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    _catalog = SkyCoord(ra=ra_all*u.deg, dec=dec_all*u.deg)
    idx, d2d, d3d = obj.match_to_catalog_sky(_catalog)
    if(d2d>0.01*u.arcsecond and d2d<2*u.arcsecond): #since there are sources with multiple epochs
        return True
    else:
        return False

def get_quality_flag(file, signal_x, signal_y, extended):
    '''
    Read in the image, return the quality flag
    '''
    quality_flag=0
    if (wash_cross(signal_x, signal_y)==False):
        quality_flag += 1
        
    black_pix_flag, _ = wash_black_pixels(file)
    if (black_pix_flag==False):
        quality_flag += 2
        
    x_size, y_size=check_image_size(file)
    if (x_size<60 or y_size<60):
        quality_flag += 10
    # if extended==True:
    #     if (wash_disturb_morph(signal_x, signal_y)==False):
    #         quality_flag += 3
    
    return quality_flag

def if_peak_centered(signal_x, signal_y, peaks_x, peaks_y, properties_x, properties_y):
    '''
    Check if the position is at the center of the radio pattern, if it is, then 2-SIDE, otherwise 1-SIDE
    '''
    # offset=10 # offset smaller than 5 is considered centered
    
    peaks_x, _ = find_peaks(signal_x, height=5, width=1, distance=10, prominence=3) # no prominence=100% peak requirement
    peaks_y, _ = find_peaks(signal_y, height=5, width=1, distance=10, prominence=3)
    
    peaks_x=np.array(peaks_x)
    peaks_y=np.array(peaks_y)
    # offset_x=np.abs(peaks_x-60) # 60 is the position in pixel of the center
    # offset_y=np.abs(peaks_y-60)
    
    if len(peaks_x)==1 and len(peaks_y)==1:
        
        #embed()
        offset_boundaries_x = properties_x['right_bases']-60 + properties_x['left_bases']-60
        offset_boundaries_y = properties_y['right_bases']-60 + properties_y['left_bases']-60
        #embed()
        # if np.min(offset_x)<offset and np.min(offset_y)<offset:
        #     return True
        if np.abs(offset_boundaries_x)<10 and np.abs(offset_boundaries_y)<10:
            return True
        else:
            return False
        
    elif len(peaks_x)>1 and len(peaks_y)==1:
        
        temp_center_x=np.average(peaks_x)
        offset_boundaries_y = properties_y['right_bases']-60 + properties_y['left_bases']-60
        
        if np.abs(temp_center_x-60)<10 and np.abs(offset_boundaries_y)<10:
            return True
        else:
            return False
        
    elif len(peaks_y)>1 and len(peaks_x)==1:
        
        temp_center_y=np.average(peaks_y)
        offset_boundaries_x = properties_x['right_bases']-60 + properties_x['left_bases']-60
        
        if np.abs(temp_center_y-60)<10 and np.abs(offset_boundaries_x)<10:
            return True
        else:
            return False
        
    elif len(peaks_y)>1 and len(peaks_x)>1:
        
        temp_center_x=np.average(peaks_x)
        temp_center_y=np.average(peaks_y)
        
        if np.abs(temp_center_x-60)<10 and np.abs(temp_center_y-60)<10:
            return True
        else:
            return False
            
    
def if_one_side(signal_x, signal_y):
    '''
    Check if separate, is it the one-side separated
    Based on if all peaks present on the only one side
    '''
    peaks_x, _ = find_peaks(signal_x, height=3, width=1, distance=5, prominence=3) # height 5->3, distance 10->5
    peaks_y, _ = find_peaks(signal_y, height=3, width=1, distance=5, prominence=3)
    
    peaks_x=np.array(peaks_x)
    peaks_y=np.array(peaks_y)
    offset_x=np.abs(peaks_x-60) # 60 is the position in pixel of the center
    offset_y=np.abs(peaks_y-60)

    try:
        center_pos_x=np.argmin(offset_x)
        center_pos_y=np.argmin(offset_y)
    except:
        return False
        
    
    # _, _, signal_x, signal_y=count_signal(file)
    # low_peaks_x, _=np.array(find_peaks(signal_x, height=3, width=2)) # select all peaks, find centered
    # low_peaks_y, _=np.array(find_peaks(signal_y, height=3, width=2))
    
    # real_center_pos_x=np.argmin(np.abs(low_peaks_x-60))
    # real_center_pos_y=np.argmin(np.abs(low_peaks_y-60))
    # embed()
    if min(offset_x)<10 and min(offset_y)<10:
        # check if the center is detected
        if (len(offset_x)>2) and len(offset_y)<=2:
            if (center_pos_x==0 or center_pos_x==len(peaks_x)-1):
                return True
            else:
                return False
        if (len(offset_y))>2 and len(offset_x)<=2:
            if (center_pos_y==0 or center_pos_y==len(peaks_y)-1):
                return True
            else:
                return False
        if ((len(offset_y))>2 and len(offset_x)>2):
            if (center_pos_y==0 or center_pos_y==len(peaks_y)-1) and (center_pos_x==0 or center_pos_x==len(peaks_x)-1):
                return True
            else:
                return False
        else:
            return True
    else:
        if((np.min(peaks_x)-60)*(np.max(peaks_x)-60))>0 and ((np.min(peaks_y)-60)*(np.max(peaks_y)-60))>0:
            # center peak is not detected, check if peaks are distributed at two sides
            return True
        else:
            return False
    

def if_offset(peaks_x, peaks_y, properties_x, properties_y, filename, ra, dec):
    '''
    Check if the source is shifted - the vlass position is not at the center
    Only for compact sources
    '''
    offset=5 # minimal offset
    max_offset=30
    cent_x=peaks_x[0]; cent_y=peaks_y[0]
    pix_center_pos=get_pix_center(filename, ra, dec, fov=120)
    cx=pix_center_pos[0]; cy=pix_center_pos[1]
    dist = (cent_x-cx)**2+(cent_y-cy)**2
    
    if (dist<offset**2 or dist>max_offset**2):
        return False
    else:
        return True
        
def get_centered_pos(properties_x, properties_y):
    '''
    Get the peaked center by left+right bases average
    '''
    cent_x = (properties_x['left_bases']+properties_x['right_bases'])/2
    cent_y = (properties_y['left_bases']+properties_y['right_bases'])/2
    
    return cent_x, cent_y

def match_to_images(ra, dec, catalog):
    '''
    Check if the entry from the table match to the downloaded VLASS images
    '''
    obj=SkyCoord(ra=str(ra), dec=str(dec))
    idx, d2d, d3d = obj.match_to_catalog_sky(catalog)
    if(d2d<2*u.arcsecond):
        return True
    else:
        return False
    
def coord_to_name(filename):
    '''
    convert a list of coordination of objects to a list of names
    '''
    table = ascii.read(filename)
    ra_col_name='RAJ2000'
    dec_col_name='DEJ2000'
    table['object_name']=ut.coord_to_name(table[ra_col_name],
                                                 table[dec_col_name],
                                                 epoch="J")
    return table

def plot_fits_image(filename):
    '''
    plot the fits file
    '''
    gc = aplpy.FITSFigure(filename)
    gc.show_greyscale()
    
def plot_single_fits_beam(fig, pos, filename, ra, dec, fov, title):
    # based on it._make_scale_bar_mult_png_axes
    # fig = plt.figure(figsize=(5,5))
    filename=folder+filename
    data, hdr, _ = it.open_image(filename, ra, dec, fov,
                                   image_folder_path=None,
                           verbosity=0)
    
    if (data is not None and hdr is not None):
        file_found = True
    else:
        file_found = False

    if file_found:
        wcs_img = wcs.WCS(hdr)

        pixcrd = wcs_img.wcs_world2pix(ra, dec, 0)
        positions = (np.float(pixcrd[0]), np.float(pixcrd[1]))
        overlap = True
        
        img_stamp = Cutout2D(data, positions, size=fov * u.arcsec,
                             wcs=wcs_img)
        cutout_positions = img_stamp.to_cutout_position((np.float(pixcrd[0]), np.float(pixcrd[1])))
        #print("VLASS cutout positions: {}".format(cutout_positions))
        
        if img_stamp is not None:

                if overlap:
                    img_stamp = img_stamp.data

                hdu = fits.ImageHDU(data=img_stamp, header=hdr)

                axs = aplpy.FITSFigure(hdu, figure=fig, subplot=pos,
                                       north=True)
                axs.axis_labels.hide()
                axs.ticks.hide()
                axs.tick_labels.hide()
                axs.add_scalebar(length=(10/3600), color='white', size=20, linewidth=3) # length has to be specified
                axs.scalebar.set_corner('top right')
                axs.scalebar.set_label('10\'\'')
                axs.scalebar.set_font(
                             weight='bold',
                             size=16)

                axs.show_markers(cutout_positions[0], cutout_positions[1]+10, layer=False, coords_frame='pixel', marker='|', c='#04F700', s=60, zorder=2)
                axs.show_markers(cutout_positions[0], cutout_positions[1]-10, layer=False, coords_frame='pixel', marker='|', c='#04F700', s=60, zorder=2)
                axs.show_markers(cutout_positions[0]+10, cutout_positions[1], layer=False, coords_frame='pixel', marker='_', c='#04F700', s=60, zorder=2)
                axs.show_markers(cutout_positions[0]-10, cutout_positions[1], layer=False, coords_frame='pixel', marker='_', c='#04F700', s=60, zorder=2)
                axs.set_title(title)
                
                axs.show_circles(cutout_positions[0], cutout_positions[1], radius=7*u.arcsec, layer=False, coords_frame='pixel', color='#66A5AD')
                axs.show_circles(cutout_positions[0], cutout_positions[1], radius=9*u.arcsec, layer=False, coords_frame='pixel', color='#66A5AD')
                
                color_map_name='inferno'

                # Sigma-clipping of the color scale
                mean = np.mean(img_stamp[~np.isnan(img_stamp)])
                std = np.std(img_stamp[~np.isnan(img_stamp)])
                try:
                    data = img_stamp[~np.isnan(img_stamp)]
                    zscale = ZScaleInterval()
                    z1, z2 = zscale.get_limits(data)
                    axs.show_colorscale(vmin=z1, vmax=z2,
                                        cmap=color_map_name)
                except:
                    # Sigma-clipping of the color scale
                    n_sigma=2
                    mean = np.mean(img_stamp[~np.isnan(img_stamp)])
                    std = np.std(img_stamp[~np.isnan(img_stamp)])
                    upp_lim = mean + n_sigma * std
                    low_lim = mean - n_sigma * std
                    axs.show_colorscale(vmin=low_lim, vmax=upp_lim,
                                        cmap=color_map_name)
                    
                axs.add_beam(hdr['BMAJ'],hdr['BMIN'],29.2995)
                axs.beam.set_color('white')
                axs.beam.set_corner('top left')
                
                                
                #fig.gca().set_title(survey + " " + band, fontweight='bold', fontsize='20')
        
def plot_single_fits_optical(fig, pos, filename, ra, dec, fov, title, vlass1_data, vlass2_data):
    # based on it._make_scale_bar_mult_png_axes
    # fig = plt.figure(figsize=(5,5))
    
    # vlass2_data, vlass_hdr=fits.getdata(vlass2_data, header=True)
    data, hdr=fits.getdata(filename, header=True) 
    
    if (data is not None and hdr is not None):
        file_found = True
    else:
        print('File/Header not found!')
        file_found = False

    if file_found:
        wcs_img = wcs.WCS(hdr)

        pixcrd = wcs_img.wcs_world2pix(ra, dec, 0)
        positions = (np.float(pixcrd[0]), np.float(pixcrd[1]))
        overlap = True
        
        img_stamp = Cutout2D(data, positions, size=fov * u.arcsec,
                             wcs=wcs_img)
        cutout_positions = img_stamp.to_cutout_position((np.float(pixcrd[0]), np.float(pixcrd[1])))
        # print("Optical cutout positions: {}".format(cutout_positions))
        
        if img_stamp is not None:

                if overlap:
                    img_stamp = img_stamp.data

                hdu = fits.ImageHDU(data=img_stamp, header=hdr)
                #vlass_hdu = fits.ImageHDU(data=vlass2_data_stamp.data, header=vlass_hdr)

                axs = aplpy.FITSFigure(hdu, figure=fig, subplot=pos,
                                       north=True)
                axs.axis_labels.hide()
                axs.ticks.hide()
                axs.tick_labels.hide()
                axs.add_scalebar(length=(10/3600), color='white', size=20, linewidth=3) # length has to be specified
                axs.scalebar.set_corner('top right')
                axs.scalebar.set_label('10\'\'')
                axs.scalebar.set_font(
                             weight='bold',
                             size=16)
                
                #axs.show_contour(vlass_hdu, levels=1, cmap='Greys')
                
                offset=10*cutout_positions[0]/59.955527780894954 # offset change from vlass image to here, get it by reading the cutout_positions
                axs.show_markers(cutout_positions[0], cutout_positions[1]+offset, layer=False, coords_frame='pixel', marker='|', c='#04F700', s=60, zorder=2)
                axs.show_markers(cutout_positions[0], cutout_positions[1]-offset, layer=False, coords_frame='pixel', marker='|', c='#04F700', s=60, zorder=2)
                axs.show_markers(cutout_positions[0]+offset, cutout_positions[1], layer=False, coords_frame='pixel', marker='_', c='#04F700', s=60, zorder=2)
                axs.show_markers(cutout_positions[0]-offset, cutout_positions[1], layer=False, coords_frame='pixel', marker='_', c='#04F700', s=60, zorder=2)
                axs.set_title(title)
                
                if vlass1_data != None:
                    #show vlass1 contour
                    vlass1_data, vlass1_hdr=fits.getdata(vlass1_data, header=True)
                    vlass1_wcs_img = wcs.WCS(vlass1_hdr)
                    
                    z1, z2=get_zscale_limits(vlass1_data)
                    vlass1_data=code_image_pos(z2, vlass1_data)
                    
                    vlass1_pixcrd = vlass1_wcs_img.wcs_world2pix(ra, dec, 0)
                    vlass1_positions = (np.float(vlass1_pixcrd[0]), np.float(vlass1_pixcrd[1]))

                    vlass1_data_stamp = Cutout2D(vlass1_data, vlass1_positions, size=fov * u.arcsec,
                                    wcs=vlass1_wcs_img)
                    vlass1_hdu = fits.ImageHDU(data=vlass1_data_stamp.data, header=vlass1_hdr)
                    axs.show_contour(vlass1_hdu, levels=1, colors='silver', linewidths=1)
                
                #show vlass2 contour
                vlass2_data, vlass2_hdr=fits.getdata(vlass2_data, header=True)
                vlass2_wcs_img = wcs.WCS(vlass2_hdr)
                
                z1, z2=get_zscale_limits(vlass2_data)
                vlass2_data=code_image_pos(z2, vlass2_data)
                
                vlass2_pixcrd = vlass2_wcs_img.wcs_world2pix(ra, dec, 0)
                vlass2_positions = (np.float(vlass2_pixcrd[0]), np.float(vlass2_pixcrd[1]))

                vlass2_data_stamp = Cutout2D(vlass2_data, vlass2_positions, size=fov * u.arcsec,
                                wcs=vlass2_wcs_img)
                vlass2_hdu = fits.ImageHDU(data=vlass2_data_stamp.data, header=vlass2_hdr)
                axs.show_contour(vlass2_hdu, levels=1, colors='white', linewidths=1)
                
                color_map_name='inferno'

                # Sigma-clipping of the color scale
                mean = np.mean(img_stamp[~np.isnan(img_stamp)])
                std = np.std(img_stamp[~np.isnan(img_stamp)])
                try:
                    data = img_stamp[~np.isnan(img_stamp)]
                    zscale = ZScaleInterval()
                    z1, z2 = zscale.get_limits(data)
                    axs.show_colorscale(vmin=z1, vmax=z2,
                                        cmap=color_map_name)
                except:
                    # Sigma-clipping of the color scale
                    n_sigma=2
                    mean = np.mean(img_stamp[~np.isnan(img_stamp)])
                    std = np.std(img_stamp[~np.isnan(img_stamp)])
                    upp_lim = mean + n_sigma * std
                    low_lim = mean - n_sigma * std
                    axs.show_colorscale(vmin=low_lim, vmax=upp_lim,
                                        cmap=color_map_name)
                
                
                                
                #fig.gca().set_title(survey + " " + band, fontweight='bold', fontsize='20')    
    
def random_sample(raw_path, sample_path, sample_size, sample_numbers):
    '''
    Make a sample of ~100 each sources (without replacement) into folders
    '''  
    
    all_filenames=read_all_files(raw_path)
    
    parent_sample_size=len(all_filenames)
    all_selected=np.array(random.sample(list(np.arange(1, parent_sample_size)), k=sample_size*sample_numbers))
    splited_selected=np.split(all_selected, sample_numbers)
    
    for sample_id, single_batch in enumerate(splited_selected):
        # put all selected files into one list
        selected_files=[]
        
        folder=str('sample'+'_'+str(sample_size)+'_'+str(sample_id+1))
        try:
            os.mkdir(sample_path+folder)
        except:
            pass
        
        print('[INFO]Generating sample No.{}'.format(sample_id+1))
        for i in single_batch:
            selected_files.append(all_filenames[i])
            
        for file in selected_files:
            # now move all files in to the folder
            if isinstance(file, str)==False:
                # multi-epoch source
                for _file in file:
                    src=raw_path+_file
                    dst=sample_path+folder+'/'
                    os.system(f"cp {src} {dst}")
            else:
                src=raw_path+file
                dst=sample_path+folder+'/'
                os.system(f"cp {src} {dst}")
    
def read_all_files(path):
    '''
    Real all files and schedule them into lists
    '''
    os.chdir(path)
    
    filenames=[]
    all_filenames=[]
    
    for file in glob.glob("*vlass*fits"):
        
        all_filename=file
        filename=file[0:20]

        filenames.append(filename)
        all_filenames.append(all_filename)
        
    _filenames=np.array(filenames); _all_filenames=np.array(all_filenames)
    filenames=[]   
    all_filenames=[] 
    # columns to write in the csv
    
    for k, filename in enumerate(_filenames):
        
        if(filename in filenames):
            #the same source in a different epoch is already registered
            continue
        
        pos=np.where(_filenames==str(filename))
        if(len(pos[0])>1):
            
            filenames.append(_filenames[k])
            all_filenames.append(_all_filenames[pos[0]])
        
        else:      
            #single-epoch sources
            filenames.append(_filenames[k])
            all_filenames.append(_all_filenames[k])
                        
    return all_filenames

if __name__ == '__main__':
    args=parse_arguments()
    global folder, output_file
    folder='/Users/xie/WORKSPACE/jet_pattern/KDEBLLACS/' #!!!put "/" at the end of the path
    #folder='/Users/xie/WORKSPACE/jet_pattern/test_single/'
    output_file='/Users/xie/WORKSPACE/jet_pattern/output/kdebllacs.csv'
    #output_file='/Users/xie/WORKSPACE/jet_pattern/output/test_single.csv'
    
    repeat_number=1 # number of artificial repeats for the signal
    
    global if_pdf
    if_pdf = args.pdf
    
    if args.mode=='single':
        plot_signal(filename=folder+args.filename, plotmode=args.plot_mode, repeat_number=repeat_number)
        
    if args.mode=='multiple':
        read_from_folder(folder)
        
    if args.mode=='count':
        count_from_folder(folder)
        
    if args.mode=='sample':
        raw_path='/Users/xie/WORKSPACE/skymapper/final_imgs/roma_bzcat/'
        sample_path='/Users/xie/WORKSPACE/jet_pattern/roma_sample/'
        random_sample(raw_path, sample_path, sample_size=100, sample_numbers=20)
    
    
    
    #print(count_signal(filename=folder+args.filename))
    
   
    
    #n_table = coord_to_name('./high_score_class_A.csv')
    
    #read_from_table(n_table, folder)
    #read_from_folder(folder)
    
