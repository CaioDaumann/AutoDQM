import json
import os
import uproot
import numpy
import awkward as ak
import os

# importing other scripts
import plot_utils

# Exploratory studies for AutoDQM - DT implementation: THIS IS OFFLINE ONLY!
#data_path = "/net/scratch_cms3a/daumann/qualification_task_muons/AutoDQM/runoffline/db/Offline/Run2022/SingleMuon/355443.root"
#data_file = uproot.open(data_path)

# For the online, the datapath is diferent!
data_path = "/net/scratch_cms3a/daumann/qualification_task_muons/AutoDQM/runoffline/db//Online/00035xxxx/0003554xx/DT/355443.root"
data_file = uproot.open(data_path)

# We have three group of histograms
problematic_histograms = []

Segment_TnP_histos       = []
Segment_TnP_name_keeper  = []

hResDist_histos          = []
hResDist_name_keeper     = []

LocalTrigger_TM_histos      = []
LocalTrigger_TM_name_keeper = []

TimeBox_histos      = []
TimeBox_name_keeper = []

for key in data_file.keys():
    if( "03-LocalTrigger-TM/Wheel1/Sector8/Station2/LocalTriggerTheta/" in key ):
        print(key)
#print( data_file["03-LocalTrigger-TM/Wheel1/Sector8/Station2/LocalTriggerPhiIn/"] )
exit()

for key in data_file.keys():
    
    if( 'Segment_TnP/Task/' in key):
        try:
            Segment_TnP_histos.append( data_file[ key ] )
            Segment_TnP_name_keeper.append( key )    
        except:
            problematic_histograms.append( key ) 
    
    if( '02-Segments' in key and 'Wheel' in key and 'Sector' in key and 'Station' in key and 'hResDist' in key ):
    
        try:
            hResDist_histos.append( data_file[ key ] )
            hResDist_name_keeper.append( key )
            
        except:
            problematic_histograms.append( key ) 

    if("03-LocalTrigger-TM" in str(key)  and "Task" in str(key)): 
        try:
            LocalTrigger_TM_histos.append( data_file[ key ] )
            LocalTrigger_TM_name_keeper.append( key )
            
            histogram = data_file[ key ] 

        except:
            problematic_histograms.append( key ) 

    # This does not exists in Record
    if("01-"in str(key) and "Digi"in str(key)): 
        try:
            TimeBox_histos.append( data_file[ key ] )
            TimeBox_name_keeper.append( key )
            
        except:
            problematic_histograms.append( key ) 

total_histos = [TimeBox_histos     , hResDist_histos,Segment_TnP_histos, LocalTrigger_TM_histos]
total_names  = [TimeBox_name_keeper,hResDist_name_keeper,Segment_TnP_name_keeper, LocalTrigger_TM_name_keeper]
folders      = ["TimeBox","hResDist","Segment_TnP", "LocalTrigger"]


test = False
"""
if( test ):
    total_histos = [LocalTrigger_TM_histos,hResDist_histos]
    total_names  = [LocalTrigger_TM_name_keeper,hResDist_name_keeper]
    folders      = ["LocalTrigger","hResDist"]    
"""

print( '\nHistograms read!\n' )
print( 'Procedding to the plot of the histograms!')

for histogram_type,histogram_names,out_folder in zip( total_histos , total_names, folders   ):

    i = 0 
    for histogram,histogram_name in zip(histogram_type,histogram_names):
        
        # The first histogram in these Local Trigger ones causes an error, and I dont know why ...
        if( out_folder == "LocalTrigger" and i > 0  ):
                plot_utils.test_2d( numpy.array(histogram.axis(0))[:,0], numpy.array(histogram.axis(1)), numpy.array(histogram.to_hist().values()), out_folder , os.path.basename(histogram_name.replace(' ', '').replace(';', '')), histogram_name.replace('DQMData/Run 355443/','').replace('DT/Run summary','') )
        else:
            a = 1+3
        i = i + 1

        #plot_utils.two_dimensional_plotter( data,  os.path.basename(histogram_name.replace(' ', '').replace(';', '')), histogram_name )
        # Snippets to the one and two dimensional plotting scripts!

        #if( out_folder == "hResDist" ):
        #    print( numpy.shape(numpy.array(data)) )
        #exit()
        if(out_folder == "hResDist"):
            data  = numpy.array( histogram.values() )
            edges = numpy.array( histogram.axis(0) )
            plot_utils. test( data, edges, os.path.basename(histogram_name.replace(' ', '').replace(';', '')), histogram_name.replace('DQMData/Run 355443/','').replace('DT/Run summary','') ,folder = out_folder )

        try:

            try:
                data  = numpy.array( histogram.values() )
                edges = numpy.array( histogram.axis(0) )
                plot_utils. test( data, edges, os.path.basename(histogram_name.replace(' ', '').replace(';', '')), histogram_name.replace('DQMData/Run 355443/','').replace('DT/Run summary','') ,folder = out_folder )
            except:
                plot_utils.test_2d( numpy.array(histogram.axis(0))[:,0], numpy.array(histogram.axis(1)), numpy.array(histogram.to_hist().values()), out_folder , os.path.basename(histogram_name.replace(' ', '').replace(';', '')), histogram_name.replace('DQMData/Run 355443/','').replace('DT/Run summary','') )
        except:
            print( 'Failed: ', histogram_name , ' Folder: ', out_folder )

        """
        try:
            #plot_utils.plotter( data,  os.path.basename(histogram_name.replace(' ', '').replace(';', '')), histogram_name.replace('DQMData/Run 355443/','').replace('DT/Run summary','') ,folder = out_folder )
            plot_utils. test( data, edges, os.path.basename(histogram_name.replace(' ', '').replace(';', '')), histogram_name.replace('DQMData/Run 355443/','').replace('DT/Run summary','') ,folder = out_folder )
        except:   
            plot_utils.two_dimensional_plotter( data,  os.path.basename(histogram_name.replace(' ', '').replace(';', '')), histogram_name.replace('DQMData/Run 355443/',''), folder = out_folder )
        """
        
"""        
Notes:
There are some histograms with fun names:
DQMData/Run 355443/JetMET/Run summary/MET/caloMet/Cleaned/<MET.meanMETTest>qr=st:100:1:MeanWithinExpected:</MET.meanMETTest>;1
That returns an error when try to read that!


"""
