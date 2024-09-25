import numpy as np

def get_odf(x, sr, odfs, hop_sec=.010, win_sec=.040 ):
    """Takes in audio, sample rate, and hop/window length. Converts hop/window length from seconds to 
    Args:
        x (vector): 1D vector contain wave
        sr (float): sample rate 
        odfs list: list of strings containing the names of the ODFs to be calculated
        hop_sec (float): hop length in seconds
        win_sec (float): window length in seconds
    """
    
    #convert hop and window length from seconds to sample
    hop_len = round(sr*hop_sec)
    win_len = int(2 ** np.ceil(np.log2(sr * win_sec)))
    half_win_len = int(win_len/2)
    
    #add padding equal to half the window length to either end of the audio file s.t. first frame is centered
    x = np.concatenate([np.zeros(half_win_len), x, np.zeros(half_win_len)])
    
    #calculate number of frames to be calculated
    frames = int(np.floor((len(x) - win_len) / hop_len + 1))
    
    
    #store onsets and var to keep prior frames value
    odf = np.zeros((len(odfs),frames))
    prior_mag = np.zeros(win_len)
    prior_2_angle = np.zeros(win_len)
    prior_angle = np.zeros(win_len)
    
    #set indices used for odf insertion
    rms_index, sf_index, hfc_index, wpd_index, cd_index, rcd_index, lfc_index = get_odf_bools(odfs)
    
    for frame_num in range(frames):
        
        #set up. FFT and compute operations on frame that are repeated
        start_ind = frame_num * hop_len
        end_ind = start_ind+win_len
        current_frame = np.fft.fft(np.multiply(x[start_ind:end_ind],
                                            np.hamming(win_len)))
        current_mag = np.abs(current_frame)
        current_angle = np.angle(current_frame)
        
        #RMS
        if rms_index >= 0 or wpd_index >= 0 :
            rms = np.sqrt(np.mean(np.power(current_mag, 2)))
            
        if rms_index >= 0:
            odf[rms_index,frame_num] = rms
        
        #SF
        if sf_index >= 0:    
            mag_dif = current_mag - prior_mag
            zero_v = np.full((mag_dif.shape[0],), 0, dtype='complex128')
            hwr = np.max((mag_dif, zero_v), axis=0)
            sf = np.mean(np.sum(hwr))
            odf[sf_index,frame_num] = sf
        
        #HFC
        if hfc_index >= 0:
            k =  list(range(half_win_len)) + list(range(half_win_len,0,-1))
            hfc = np.mean(np.power(current_mag, 2) * np.abs(k))
            odf[hfc_index,frame_num] = hfc
            
        #LFC
        if lfc_index >= 0:
            k =  [0] + list(range(half_win_len,0,-1)) + list(range(2, half_win_len +1))
            lfc = np.mean(np.power(current_mag, 2) * np.abs(k))
            odf[lfc_index,frame_num] = lfc
        
        #WPD
        #not using phase decay as it performs worse in dixon 2006,
        phase_der_2 = np.subtract(current_angle, np.subtract(np.multiply(prior_angle, 2), prior_2_angle))
        pd_summands = np.abs(np.divide(np.subtract(np.mod(np.add(phase_der_2, np.pi), 2 * np.pi), np.pi), np.pi))
        
        if wpd_index >= 0:
            if rms != 0:
                wpd = np.divide(np.mean(np.multiply(pd_summands, current_mag)), rms * 2)
            else:
                wpd = 0
            odf[wpd_index,frame_num] = wpd
        
        #CD
        #Note: this code is taken from the labs and tutorial session solutions. 
        #It was used in evaluation, but not in the final system, so I have not changed it.
        if cd_index >= 0 or rcd_index >= 0:
            cd_summands = np.sqrt(np.subtract(np.add(np.power(prior_mag,2), np.power(current_mag,2)),
                                np.multiply(np.multiply(np.multiply(prior_mag, current_mag), 2),
                                            np.cos(np.subtract(current_angle, phase_der_2)))))
        if cd_index >= 0:
            cd = np.mean(cd_summands)
            odf[cd_index,frame_num] = cd
        
        #RCD
        #Note: this code is taken from the labs and tutorial session solutions. 
        #It was used in evaluation, but not in the final system, so I have not changed it.
        if rcd_index >= 0:
            rcd = np.mean(np.multiply(np.greater_equal(current_mag, prior_mag), cd_summands))
            odf[rcd_index,frame_num] = rcd
        
        #update stored values
        prior_mag = current_mag
        prior_2_angle = prior_angle
        prior_angle = current_angle
        
    return odf



def get_odf_bools(odfs): 
    """Used to determine which position an odf should be inserted. 
    Holdover from analysis when multiple odfs were tested at once. 
    User provides list of odfs and function calculates position in array

    Args:
        odfs (_type_): list of strings with the odfs to be calcuated

    Returns:
        scalar: each var is a index corresponding to insert position in output of OnsetDetector
    """
    if 'RMS' in odfs:
        rms_index = odfs.index('RMS')
    else:
        rms_index = -1
        
    if 'SF' in odfs:
        sf_index = odfs.index('SF')
    else:
        sf_index = -1
        
    if 'HFC' in odfs:
        hfc_index = odfs.index('HFC')
    else:
        hfc_index = -1

    if 'WPD' in odfs:
        wpd_index = odfs.index('WPD')
    else:
        wpd_index = -1
    
    if 'CD' in odfs:
        cd_index = odfs.index('CD')
    else:
        cd_index = -1

    if 'RCD' in odfs:
        rcd_index = odfs.index('RCD')
    else:
        rcd_index = -1
    
    if 'LFC' in odfs:
        lfc_index = odfs.index('LFC')
    else:
        lfc_index = -1
    
    return rms_index, sf_index, hfc_index, wpd_index, cd_index, rcd_index, lfc_index


def median_filt(odf,wd):
    """median filter applied to each row of array x to get a (filt_len,odf_len) dimensional matrix.
    Iterate through rows, apply the filter, take the median
   endpoints are repeated to fill the amount each row has been shifted by.

    Args:
        odf (vector): two diemnsional vector containing odf values (rows are odfs, columns are frames)
        wd (int): value specifying the length of the median filter

    Returns:
        vector: two dimensional vector of median filtered ODF values
    """
    
    #Account for case when using one odf
    if len(odf.shape) == 1:
        odf = np.expand_dims(odf, axis = 0)
    
    odf_count = odf.shape[0]
    frames = odf.shape[1]
    odf_filt = np.zeros((odf_count,frames))
    #index in the middle of the middle of the filter length
    mid_index = (wd - 1) // 2
    
    for odf_ind in range(odf_count):
        #create filt_length entries for each value in odf vector
        y = np.zeros((wd, frames), dtype=odf.dtype)
        odf_vals = odf[odf_ind,:]
        #set middle index equal to odf values
        y[mid_index,:] = odf_vals
        #fill
        for lower_row_ind in range(mid_index):
            upper_row_ind = wd - lower_row_ind - 1
            
            #shift forwards by mid_index - lower_row_ind
            shift_index = mid_index - lower_row_ind
            y[lower_row_ind,shift_index:] = odf_vals[:-shift_index]
            y[lower_row_ind,:shift_index] = odf_vals[0]
            
            #shift backwards by mid_index - lower_row_ind
            y[upper_row_ind, :-shift_index] = odf_vals[shift_index:]
            y[upper_row_ind, -shift_index: ] = odf_vals[-1]
            
        odf_filt[odf_ind,:] = np.median(y, axis=0)
        
    return odf_filt