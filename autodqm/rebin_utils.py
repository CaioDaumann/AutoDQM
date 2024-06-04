import uproot
import numpy as np

import os

#import rebinning_studies_utils.plot_rebinned_histo as prh

def Rebin(hist, rbX, rbY, pull = False):
        
        # Get the dimensions of the input histogram
        x_bins, y_bins = hist.shape
        
        # Check if dimensions are divisible by the rebinning factors
        #if x_bins % rbX != 0 or y_bins % rbY != 0:
        #    raise ValueError("Histogram dimensions are not divisible by the rebinning factors.")
        
        # Calculate the dimensions of the rebinned histogram
        new_x_bins = x_bins // rbX
        new_y_bins = y_bins // rbY
        
        # Initialize the rebinned histogram
        rebinned_hist = np.zeros((new_x_bins, new_y_bins))
        
        # Loop over the new bins and sum the appropriate values
        for i in range(new_x_bins):
            for j in range(new_y_bins):
                # Sum the values in the current rebinning block
                if(pull):
                    rebinned_hist[i, j] = hist[i*rbX:(i+1)*rbX, j*rbY:(j+1)*rbY].sum()/(rbX*rbY)
                else:
                    rebinned_hist[i, j] = int(hist[i*rbX:(i+1)*rbX, j*rbY:(j+1)*rbY].sum()/(rbX*rbY))
        
        return rebinned_hist     
    
def pad_histogram(histo):

    # Calculate the padding needed to make the first dimension non-prime
    new_x_bins = histo.shape[0] + 1  # Next even number after 73
    pad_width = new_x_bins - histo.shape[0]

    # Pad the histogram along the first dimension with zeros
    padded_hist = np.pad(histo, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
    
    return padded_hist
    
# Instead of rebinning the data and reference histograms, we will rebin the pull histogram
def rebin_pull_hist(pull_hist, data_hist_raw,ref_hists_raw, ref_hist_sum, tol, histpair):
   
        import plugins.beta_binomial as bb
   
        ## only filled bins used for chi2
        nBinsUsed = np.count_nonzero(np.add(ref_hist_sum, data_hist_raw))
        nBins = data_hist_raw.size   

        # Histograms are paddaded to allow for rebinning
        data_hist_raw = pad_histogram(data_hist_raw)
        ref_hists_raw = [pad_histogram(rhr) for rhr in ref_hists_raw]  
             
        nBinsX, nBinsY = np.shape(data_hist_raw)[0], np.shape(data_hist_raw)[1]
        
        # After the padding we need to re-calculate the pull histogram
        [pull_hist, ref_hist_prob_wgt] = bb.pull(data_hist_raw, ref_hists_raw, tol)
        pull_hist = pull_hist*np.sign(data_hist_raw-ref_hist_prob_wgt)
        
        chi2_keep = []
        maxpull_keep = []
        for rbX in range(1, 5): 
            ## Do a few small rebinnings, but only do large bins if nBinsX divides evenly into rbX
            if rbX > 5 and (nBinsX % rbX != 0): break
            for rbY in range(1, 5):       
        
                pull_hist_rb = Rebin(pull_hist, rbX, rbY, pull = True)
                
                # Number of bins in the new histogram
                nBinsUsed_rb = np.count_nonzero(pull_hist_rb)
                
                # Calculate the chi2 and MaxPull value for the rebinned histogram
                chi2     = np.sqrt(np.count_nonzero(pull_hist)/nBinsUsed_rb)*np.square(pull_hist_rb).sum() / nBinsUsed_rb
                max_pull = bb.maxPullNorm(np.amax(pull_hist_rb), nBinsUsed_rb)
                min_pull = bb.maxPullNorm(np.amin(pull_hist_rb), nBinsUsed_rb)
                if abs(min_pull) > max_pull:
                    max_pull = min_pull  
                
                # Plots the rebin version of the histogram. Very expensive to do it for all histograms
                #prh.plot_rebinned_histo(pull_hist_rb, rebin_x = rbX, rebin_y = rbY, chi2 = chi2, maxpull = max_pull, name = histpair.data_name )        
        
                chi2_keep.append(chi2)
                maxpull_keep.append(max_pull)
                
        return np.max(chi2_keep), np.max(maxpull_keep)