import numpy as np
from .odf import get_odf, median_filt
from .utilities import evaluate_results


class OnsetDetector:
    """Class that produces onsets and variations for N number of onset dectection functions. 
        Class was created when multiple onset detection functions were being tested, so it is generalized to
        to handle two dimensional arrays of onsets. assumes the rows are odfs and columns are frames
    Args:
        x np.array: audio signal
        sr scalar: sample rate of audio signal
        onset_param_dict dict: dictionary of onset detection parameters
    """
    def __init__(self, x, sr, onset_param_dict):
        
        #Input values
        self.x = x
        self.sr = sr
        self.odf_names = onset_param_dict['odfs'] #list of odf names to be used
        self.num_odfs = len(self.odf_names)
        self.frame_count = None
        self.t_secs = None
        
        #parameters
        self.win_sec = onset_param_dict['win_sec']
        self.hop_sec = onset_param_dict['hop_sec']
        self.w = onset_param_dict['w']
        self.m = onset_param_dict['m']
        self.lam = onset_param_dict['lam']
        self.al = onset_param_dict['al']
        self.wd = onset_param_dict['wd']
        
        #crated by callimg get_odf(), get_odf_med(), get_odf_norm() in sequence
        self.odf = None
        self.odf_med = None
        self.odf_norm = None
        
        #Results for each ODF passed into the object. 
        self.peak_bools = None # self.num_odfs x self.frame_count array of boolean values representing peaks
        self.peak_time_array = [] #self.num_odfs length array of np.array with peak times
        self.ioi_array = [] #self.num_odfs length array of np.array with for each set of peak times
        self.peak_odf_array = [] #self.num_odfs length array of ODF values corresponding to peak times in self.peak_time_array 
        self.peak_odf_norm_array = [] # self.peak_odf_array , but normalized values
        self.peak_index_array = [] # self.num_odfs length array of np arrays with indices linking onsets to odf frame index
        
        #additional Results defined when there is only one ODF name is passed into the object. Each is a 1d np.array
        self.peak_times = None
        self.iois = None
        self.peak_odf_vals = None
        self.peak_odf_norm_vals = None
        self.peak_index = None
        self.eval_dict = None #dictionary of evaluation scores for the onset detection
    
    #Set odf array: MxN array of onsets where M is the number of ODFs and N is the number of onsets
    def set_odf(self):
        self.odf = get_odf(self.x, self.sr, self.odf_names, self.hop_sec, self.win_sec)
        self.frame_count = self.odf.shape[1]
        self.t_secs = np.multiply(range(self.frame_count), self.hop_sec)
        self.frame_secs = self.t_secs[1]
    
    #apply median filter to odf array
    def set_odf_med(self):
        self.odf_med = median_filt(self.odf, self.wd)
        
    #normalize the MxN array of onsets across each row. Produces MxN array
    def set_odf_norm(self):
        odf_means = np.expand_dims(np.mean(self.odf_med, axis= 1),axis=1)
        odf_stds = np.expand_dims(np.mean(self.odf_med, axis=1),axis=1)
        self.odf_norm = np.divide(np.subtract(self.odf_med, odf_means), odf_stds)
    
    #applies the three conditions detailed in dixon, 2006 to get MxN array of 1/0 values marking onsets
    def set_peak_bools(self, odf):
        odf_shape = self.odf.shape
        
        #pad the odf with zeros to center the frame, and then create empty arrays to store boolean values for each condition
        odf_padding = np.concatenate([np.zeros((self.num_odfs,self.w*self.m)), odf ,np.zeros((self.num_odfs, self.w))],axis=1)
        odf_cond1 = np.zeros(odf_shape)
        odf_cond2 = np.zeros(odf_shape)
        odf_cond3 = np.zeros(odf_shape)
        
        #initialized values stored for prior frames needed in contiion three
        prior_prior_g_n = np.zeros(self.num_odfs)
        prior_f_n = np.zeros(self.num_odfs)
        
        #iterate through each frame and check each condition
        for frame_ind in range(self.w* self.m, self.frame_count + self.w*self.m ):
            ins_col = frame_ind - self.w*self.m
            f_n = odf_padding[:,frame_ind]
            
            odf_slice_cond1 = odf_padding[:,frame_ind-self.w:frame_ind+self.w+1]
            odf_slice_cond2 = odf_padding[:,frame_ind-self.w*self.m:frame_ind+self.w+1]
            
            #1: f(n) >= f(k) for all k s.t. n-w <= n <= n+w
            odf_cond1[:,ins_col] = np.greater_equal(f_n, np.max(odf_slice_cond1, axis=1))
            
            #2: from k= n - m*w to k = m + w, sum f(k)/(m*w + w + 1 ) + lambda
            odf_cond2[:,ins_col] = np.greater_equal(f_n, np.sum(odf_slice_cond2, axis=1)/(self.m*self.w+self.w+1) + self.lam ) 
            
             #3: f(n) >= g(n-1) = max(f(n-1) , alpha * g(n-2) + (1-alpha)*f(n-1) )
            prior_g_n = np.maximum(prior_f_n, self.al*prior_prior_g_n + (1-self.al)*prior_f_n)
            odf_cond3[:,ins_col] = np.greater_equal(f_n,prior_g_n)
            
            prior_f_n = f_n
            prior_prior_g_n = prior_g_n
        
        #set instance var equal to self.num_odfs x self.frame_count array of boolean values representing peaks
        self.peak_bools =  odf_cond1 * odf_cond2 * odf_cond3
    
    #applies full peak picking algorithm by calculating odf, smooths, normalizes, find peaks, and then stores the results
    def set_onsets(self):
        #assigns odf values
        self.set_odf()
        
        #apply median filter to odf array (not used with best performing parameters)
        self.set_odf_med()
        
        #normalize the MxN array of onsets across each row
        self.set_odf_norm()
        
        #set array with booleans marking peaks
        self.set_peak_bools(self.odf_norm)
        
        #for each odf, find the peaks and store the results
        for i in range(0, self.num_odfs):
            
            #gets indices of non zero boleans for odf i
            peak_index = np.nonzero(self.peak_bools[i,:])[0]
            
            #gets values for odf i
            odf_vals = self.odf[i,:]
            odf_norm_vals = self.odf_norm[i,:]
        
            #sleects peak times, peak odf values, and peak values from normalized odf
            peak_times = self.t_secs[peak_index]
            peak_odf = odf_vals[peak_index]
            peak_odf_norm = odf_norm_vals[peak_index]
            
            #use broadcasting to compute matrix of parwise differences between elements in vector
            #only want non zero elements in the lower triangle as the differences are mirrored along the diagonal (but negative)
            onset_difs = peak_times.reshape(-1, 1) - peak_times
            comp_ioi_tri = np.tril(onset_difs, k=-1)
            comp_iois = comp_ioi_tri[comp_ioi_tri != 0]
            
            self.peak_time_array.append(peak_times)
            self.ioi_array.append(comp_iois)
            self.peak_odf_array.append(peak_odf)
            self.peak_odf_norm_array.append(peak_odf_norm)
            self.peak_index_array.append(peak_index)
        
        #if input is only one odf, assign to 1d instance variables
        if self.num_odfs == 1:
            self.peak_times = peak_times
            self.iois = comp_iois
            self.peak_odf_vals = peak_odf
            self.peak_odf_norm_vals = peak_odf_norm
            self.peak_index = peak_index
            self.odf = self.odf[0] #if only one odf, then convert shape from 1,N to N
            self.odf_norm = self.odf_norm[0] #if only one odf, then convert shape from 1,N to N
    
    def eval_onsets(self, annotations):
        self.eval_dict = evaluate_results(self.peak_times, annotations)
    
    def print_eval(self):
        print('Onset Evaluation')
        print('ODF: {0} \n F Score: {1}  \n P Score: {2}'.format(self.odf_names[0], 
                                                                 self.eval_dict['f_measure'], 
                                                                 self.eval_dict['p_score'] ))
    
    #get methods to access the results
    def get_odf(self):
        return self.odf
    
    def get_eval(self):
        return self.eval_dict

    def get_peak_times(self):
        return self.peak_times
    
    def get_iois(self):
        return self.iois
    
    def get_peak_odf(self):
        return self.peak_odf_vals
    
    def get_peak_odf_norm(self):
        return self.peak_odf_norm_vals
    
    def get_peak_indexs(self):
        return self.peak_index
    
    def get_frame_secs(self):
        return self.hop_sec
    
    def get_odf_norm(self):
        return self.odf_norm
    