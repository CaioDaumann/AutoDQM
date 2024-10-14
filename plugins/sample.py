import uproot
import numpy as np
from autodqm.plugin_results import PluginResults
from autodqm.rebin_utils import rebin_pull_hist
import autodqm.rebin_utils as rebin_utils
import scipy.stats as stats
from scipy.special import gammaln
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plugins.pullvals import normalize_rows

import onnxruntime as ort

import os

def comparators():
    return {
    'sample' : sample
    }


def sample(histpair, **kwargs):
    
    data_hist_orig = histpair.data_hist
    ref_hists_orig = [rh for rh in histpair.ref_hists if rh.values().size == data_hist_orig.values().size]
    
    data_hist_raw = np.round(np.copy(np.float64(data_hist_orig.values())))
    ref_hists_raw = np.round(np.array([np.copy(np.float64(rh.values())) for rh in ref_hists_orig]))
    
    ## Delete empty reference histograms
    ref_hists_raw = np.array([rhr for rhr in ref_hists_raw if np.sum(rhr) > 0])
    nRef = len(ref_hists_raw)
    
    ## num entries
    data_hist_Entries = np.sum(data_hist_raw)
    ref_hist_Entries = [np.sum(rh) for rh in ref_hists_raw]
    ref_hist_Entries_avg = np.round(np.sum(ref_hist_Entries) / nRef)    
    
    ############################################################
    # Lets pre-process the data_hist_orig for the autoencoder
    
    # Lets extract the conditions (wheel, sector, station)
    conditions = np.array(extract_numbers(histpair.data_name)).reshape(1,3).astype(np.float32)
    
    # Now we apply the hot bin algorithm to the data_hist_orig
    histogram = rebin_utils.substitute_max_bin_with_average(data_hist_raw)
    
    # Now we pad the histogram
    padded_histogram = pad_array(histogram, 100)
    padded_histogram = padded_histogram.reshape(1,1, 100, 12).astype(np.float32)
    
    # Now we pre-process the histogram (divide by the max of the histogram)
    padded_histogram = padded_histogram / np.max(padded_histogram, axis = (2,3) , keepdims = True)
    
    ####################################
    # Now we start the onnx session
    # perhaps this should be done outside of the histograms loop? - How much time would that bring?
    
    # Load the ONNX model
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_model_path = os.path.join(script_dir, "../models/autoencoder_dt/linear_embedding_conv_autoencoder_test_version_fixed.onnx")
    session = ort.InferenceSession(onnx_model_path)   
    #onnx_model_path = "./models/autoencoder_dt/linear_embedding_conv_autoencoder.onnx"
    #session = ort.InferenceSession(onnx_model_path)
    
    # Get the input and output names for the ONNX model
    input_name = session.get_inputs()[0].name
    conditions_name = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    inputs = {input_name: padded_histogram, conditions_name: conditions}
    output = session.run([output_name], inputs)
    output = output[0].squeeze(1)
    
    sse = np.sqrt(np.sum( histogram[histogram > 0] ))*np.sum( (padded_histogram.squeeze(1) - output[0])**2 )
    nEntries = np.sum( histogram[histogram > 0] )
    
    # Histogram that highlights the anomalies
    anomaly_map_histogram = np.square(padded_histogram - output[0])[0][0]
    
    # removing the padding for plotting!
    anomaly_map_histogram = anomaly_map_histogram[:data_hist_raw.shape[0], :] 
    
    #print(anomaly_map_histogram   )
    #print( anomaly_map_histogram.shape )
    #exit()
    
    # Threhsold - add this to json afterwards
    sse_threhsold = 1725
    
    ## define if plot anomalous
    is_outlier = sse > sse_threhsold
    
    #########################################
    ####### Plotting starts
    #########################################
    
    ## Summed ref_hist
    ref_hist_sum = ref_hists_raw.sum(axis=0)
    
    ## only filled bins used for chi2
    nBinsUsed = np.count_nonzero(np.add(ref_hist_sum, data_hist_raw))
    nBins = data_hist_raw.size
    
    ## plotting!
    # For 1D histograms, set pulls larger than pull_cap to pull_cap
    if data_hist_raw.ndim == 1:
        pull_hist = np.where(pull_hist >  pull_cap,  pull_cap, pull_hist)
        pull_hist = np.where(pull_hist < -pull_cap, -pull_cap, pull_hist)
    # For 2D histograms, set empty bins to be blank
    if data_hist_raw.ndim == 2:
        anomaly_map_histogram = np.where(np.add(ref_hist_sum, data_hist_raw) == 0, None, anomaly_map_histogram)

    if nRef == 1:
        ref_runs_str = histpair.ref_runs[0]
    else:
        ref_runs_str = str(min([int(x) for x in histpair.ref_runs])) + ' - '
        ref_runs_str += str(max([int(x) for x in histpair.ref_runs]))
        ref_runs_str += ' (' + str(nRef) + ')'    
    
    # Now plotting! the anomaly map === borrowing functuon from the beta_binomial function
    ##---------- 2d Plotting --------------
    # Check that the hists are 2 dimensional
    if ( (       "TH2" in str(type(data_hist_orig)) and       "TH2" in str(type(ref_hists_orig[0])) ) or
         ("TProfile2D" in str(type(data_hist_orig)) and "TProfile2" in str(type(ref_hists_orig[0])) ) ):
        
        colors = ['rgb(215, 226, 194)', 'rgb(212, 190, 109)', 'rgb(188, 76, 38)']
        #Getting Plot labels for x-axis and y-axis as well as type (linear or categorical)
        xLabels = None
        yLabels = None
        c = None
        x_axis_type = 'linear'
        y_axis_type = 'linear'
        if data_hist_orig.axes[0].labels():
           xLabels = [str(x) for x in data_hist_orig.axes[0].labels()]
           x_axis_type = 'category'
        else:
           xLabels = [ str( data_hist_orig.axes[0]._members["fXmin"] +
                      x * ( data_hist_orig.axes[0]._members["fXmax"] -
                            data_hist_orig.axes[0]._members["fXmin"] ) /
                            data_hist_orig.axes[0]._members["fNbins"] )
                       for x in range(0, data_hist_orig.axes[0]._members["fNbins"] + 1) ]

        if data_hist_orig.axes[1].labels():
           yLabels = [str(x) for x in data_hist_orig.axes[1].labels()]
           y_axis_type = 'category'
        else:
           yLabels = [ str(data_hist_orig.axes[1]._members["fXmin"] +
                      x * (data_hist_orig.axes[1]._members["fXmax"] -
                           data_hist_orig.axes[1]._members["fXmin"] ) /
                           data_hist_orig.axes[1]._members["fNbins"] )
                       for x in range(0, data_hist_orig.axes[1]._members["fNbins"] + 1) ]
    
        if("xlabels" in histpair.config.keys()):
            xLabels=histpair.config["xlabels"]
            x_axis_type = 'category'
        if("ylabels" in histpair.config.keys()):
            yLabels=histpair.config["ylabels"]
            y_axis_type = 'category'

        anomaly_map_histogram = np.transpose(anomaly_map_histogram)
    
        #Getting Plot Titles for histogram, x-axis and y-axis
        xAxisTitle = data_hist_orig.axes[0]._bases[0]._members["fTitle"]
        yAxisTitle = data_hist_orig.axes[1]._bases[0]._members["fTitle"]
        plotTitle = histpair.data_name + " beta-binomial  |  data:" + str(histpair.data_run) + " & ref:" + ref_runs_str
    
        #Plotly doesn't support #circ, #theta, #phi but does support unicode
        xAxisTitle = xAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
        yAxisTitle = yAxisTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
        plotTitle = plotTitle.replace("#circ", "\u00B0").replace("#theta","\u03B8").replace("#phi","\u03C6").replace("#eta","\u03B7")
    
        #Plot pull-values using 2d heatmap will settings to look similar to old Pyroot version
        c = go.Figure(data=go.Heatmap(z=anomaly_map_histogram, zmin=0, zmax=0.4, colorscale=colors, x=xLabels, y=yLabels))
        c['layout'].update(plot_bgcolor='white')
        c.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=False, type=x_axis_type)
        c.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=False, type=y_axis_type)
        c.update_layout(
            title=plotTitle , title_x=0.5,
            xaxis_title= xAxisTitle,
            yaxis_title= yAxisTitle,
            font=dict(
                family="Times New Roman",
                size=9,
                color="black"
            )
        )
    ##----- end 2D plotting! --------
    
    if nRef == 1:
        Ref_Entries_str = str(int(ref_hist_Entries[0]))
    else:
        Ref_Entries_str = " - ".join([str(int(min(ref_hist_Entries))), str(int(max(ref_hist_Entries)))])

    info = {
        'Chi_Squared': float(round(sse, 2)),
        'Max_Pull_Val': float(round(sse,2)),
        'SSE_score': float(round(sse,2)),
        'Data_Entries': str(int(data_hist_Entries)),
        'Ref_Entries': Ref_Entries_str
    }

    #print( np.min( anomaly_map_histogram ), np.max(anomaly_map_histogram) )

    artifacts = [anomaly_map_histogram, str(int(data_hist_Entries)), Ref_Entries_str]    
    
    #print(  )
    
    return PluginResults(
        c,
        show=bool(is_outlier),
        info=info,
        artifacts=artifacts)
    

def extract_numbers(input_string):
    parts = input_string.split('_')
    w_value = int([part[1:] for part in parts if part.startswith('W')][0])
    st_value = int([part[2:] for part in parts if part.startswith('St')][0])
    sec_value = int([part[3:] for part in parts if part.startswith('Sec')][0])
    return (w_value, st_value, sec_value)

def pad_array(array, max_length):
    padding_length = max_length - array.shape[0]
    array_mean = np.mean(array)
    array_std = np.std(array)
    if padding_length > 0:
        padding = np.zeros((padding_length, array.shape[1])) 
        padded_array = np.vstack((array, padding))
        padded_array = np.nan_to_num(padded_array, nan=array_mean)
        return padded_array
    return array
