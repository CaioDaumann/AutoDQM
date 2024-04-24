import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Import tqdm for the progress bar
import shutil
import random
import pandas as pd

# Importing other scripts
import plot_utils

# Define a list with suitable runs to be used as reference in the meta-studies!
available_ref_runs = [  "355872","355892","355912","355913","355921","355988","356005","356043","356076","356077","356309","356316","356323","356378","356381","356383","356386","356433","356446","356523","356531",
                        "356563","356568","356569","356570","356576","356578","356580","356582","356615","356619","356812","356814","356824","356908","356919","356937","356946","356947","356948","356951","356954",
                        "356955","356956","356968","356969","356970","356999","357000","357001","357079","357080","357081","357106","357112","357268","357271","357328","357329","357330","357331","357332","357333",
                        "357401","357406","357438","357440","357441","357442","357472","357479","357482","357538","357542","357550","357610","357611","357612","357613","357688","357696","357698","357699","357700",
                        "357701","357705","357706","357732","357734","357735","357754","357756","357758","357759","357777","357778","357779","357802","357803","357804","357805","357807","357808","357809","357812",
                        "357813","357814","357815","357898","357899","357900","359602","359661","359685","359686","359693","359694","359699","359718","359751","359762","359763","359764","359806","359808","359809",
                        "359810","359812","359814","359871","359998","360017","360019","360075","360090","360116","360125","360126","360127","360128","360131","360141","360224","360225","360295","360296","360327",
                        "360392","360393","360400","360413","360428","360437","360459","360460","360490","360491","360761","360794","360795","360820","360825","360826","360856","360874","360876","360887","360888",
                        "360889","360890","360892","360895","360919","360921","360927","360941","360942","360945","360946","360948","360950","360951","360991","360992","361020","361044","361045","361052","361054",
                        "361083","361091","361105","361106","361107","361110","361188","361193","361197","361223","361239","361240","361272","361280","361284","361297","361303","361318","361320","361333","361361",
                        "361362","361363","361365","361366","361400","361417","361443","361468","361475","361512","361569","361573","361579","361580","361957","361971","361989","361990","361994","362058","362059",
                        "362060","362061","362062","362087","362091","362104","362105","362106","362107","362148","362153","362154","362161","362163","362166","362167","362597","362614","362615","362616","362617",
                        "362618","362653","362654","362655","362657","362695","362696","362698","362720","362728","362757","362758","362760"]

# We have to separate things for the offline studies into SingleMuon and Muon runs!
# lots of available muon runs!
available_muon_ref_runs = ["356563","356568","356569","356570","356576","356578","356580","356582","356615","356619","356812","356814","356824","356908","356919","356937","356946","356947","356948","356951","356954",
                "356955","356956","356968","356969","356970","356999","357000","357001","357079","357080","357081","357106","357112","357268","357271","357328","357329","357330","357331","357332","357333",
                "357401","357406","357438","357440","357441","357442","357472","357479","357482","357538","357542","357550","357610","357611","357612","357613","357688","357696","357698","357699","357700",
                "357701","357705","357706","357732","357734","357735","357754","357756","357758","357759","357777","357778","357779","357802","357803","357804","357805","357807","357808","357809","357812",
                "357813","357814","357815","357898","357899","357900","359602","359661","359685","359686","359693","359694","359699","359718","359751","359762","359763","359764","359806","359808","359809",
                "359810","359812","359814","359871","359998","360017","360019","360075","360090","360116","360125","360126","360127","360128","360131","360141","360224","360225","360295","360296","360327"]

############################################################################################################################################################################

# Very few available SingleMuon runs ... , why is that?
available_SingleMuon_ref_runs = ["355872","355892","355912","355913","355921","355988","356005","356043","356076","356077","356309","356316","356323","356378","356381","356383","356386"]

def process_run(hist_type, data_run, bad_runs, debug, rRefRuns, online,threshold=10):

    # Reference runs taken from: https://docs.google.com/spreadsheets/d/1tzRXeT9Y46Ke0V1tB4H9ZN_xftCe8xRd/edit#gid=1574718651
    # We select only the runs labeled as "good" to train in the spreadsheet
 
    if(online):
        
        # "Sampling" the reference runs from the available ones defined above in the code
        ref_runs = random.sample(available_ref_runs, rRefRuns)
        ref_runs = '_'.join(map(str, ref_runs))
        
        # For online we have to use the data_ref_series and data_ref_sample to be able to run the code. We extract it in the two line below
        data_ref_series = '000' + str(data_run)[:2] + 'xxxx'
        data_ref_sample = '000' + str(data_run)[:4] + 'xx'
        
        # Command to run the ./run-offline.py script and perform the meta-studies
        cmd = f"./run-offline.py Online {hist_type} {data_ref_series} {data_ref_sample} {data_run} {ref_runs}  --ref_series 00035xxxx --ref_sample 0003551xx --threshold {threshold}"

        if(debug):
            subprocess.run(cmd, shell=True, cwd="../runoffline/")
        else:
            subprocess.run(cmd, stdout=subprocess.DEVNULL,shell=True, cwd="../runoffline/")
        #print(cmd)
    else:
        
        # For offline (DT DOC3) the 2022 runs are separated into SingleMuon and Muon runs, so we have to select the reference runs accordingly
        data_ref_series = "Run2022"
        data_ref_sample = "SingleMuon" 

        # Example on how to run a 2023 run below
        # To run with the 2023 data ->  cmd = f"./run-offline.py Offline {hist_type} Run2023 SingleMuon {data_run} 355892_355912_355913_355921_356005_356076_356077_356309 --ref_series --ref_series Run2022 --ref_sample Muon  "

        try:
            
            # "Sampling" the reference runs from the available ones defined above in the code
            ref_runs = random.sample(available_SingleMuon_ref_runs, rRefRuns)
            ref_runs = '_'.join(map(str, ref_runs))
            
            cmd = f"./run-offline.py Offline {hist_type} Run2022 SingleMuon {data_run} {ref_runs} --ref_series Run2022 --ref_sample SingleMuon  " #355135_355134_355872_359814_359764"
            
            if(debug):
                result = subprocess.run(cmd, check=True, shell=True, cwd="../runoffline/")
            else:
                result = subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,shell=True, cwd="../runoffline/")
        
        except subprocess.CalledProcessError as e:
            
            # "Sampling" the reference runs from the available ones defined above in the code
            ref_runs = random.sample(available_muon_ref_runs, rRefRuns)
            ref_runs = '_'.join(map(str, ref_runs))
            
            cmd = f"./run-offline.py Offline {hist_type} Run2022 Muon {data_run} {ref_runs} --ref_series Run2022 --ref_sample Muon  " #355135_355134_355872_356321_356071_355205_352503_353014 #355135_355134_355872_359814_359764"
            
            if(debug):
                result = subprocess.run(cmd, shell=True, cwd="../runoffline/")
            else:
                result = subprocess.run(cmd, stdout=subprocess.DEVNULL,shell=True, cwd="../runoffline/")            

    # Now, hopefully the run is done, and we can read outputs of the run offline script
    # It outputs the PDFs (if the histogram is flagged as anomalous) and the max pull and chi2 values for each histogram in a .txt format
    pdf_folder = f'../runoffline/out/pdfs/{hist_type}/{data_run}'
    
    pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith('.pdf')]
    num_pdfs = len(pdf_files)
    
    # Reading the max pull values
    txt_files = [file for file in os.listdir(pdf_folder) if file.endswith('pull.txt')]
    max_pull_values = []
    for txt_file in txt_files:
        with open(pdf_folder + '/' + txt_file, 'r') as file:
            # Read the content of the file
            content = file.read()
            max_pull_values.append(abs(float(content)))        

    for txt_file in txt_files:
        os.remove(pdf_folder + '/' + txt_file)

    # Reading the max Chi2 of the histograms
    txt_files = [file for file in os.listdir(pdf_folder) if file.endswith('Chi2.txt')]
    max_chi2_values = []
    for txt_file in txt_files:
        with open(pdf_folder + '/' + txt_file, 'r') as file:
            # Read the content of the file
            content = file.read()
            max_chi2_values.append(abs(float(content)))  

    # Now we exclude the pdfs and txt so we dont double count next run!
    for pdf_file in pdf_files:
        os.remove(os.path.join(pdf_folder, pdf_file))
    
    for txt_file in txt_files:
        os.remove(pdf_folder + '/' + txt_file)

    is_bad_run = data_run in bad_runs

    return hist_type, data_run, num_pdfs, is_bad_run, max_pull_values, max_chi2_values

def main():
    
    print('Starting the meta-studies for the 2022 runs!\n')
    print('Remember to set the environment variables manually!')
    
    ##############################################################################################################
    # Setup variables
    debug = False
    online = True
    
    # Make a json to all of this information! -----------
    
    # Array containing the number of reference runs to be use for the meta-studies and threshold testing!
    n_ref_runs = [8,4,2,1]
    
    # Since things get a bit more unstable with lesses ref runs, we ran multiple times and average the thresholds!
    n_trials = [5,6,6,6]
    
    # Thresholds to be used in the meta-studies
    metastudies_thresholds = [0.95,0.8]
    
    histogram_type = ["DOC1_DT_Timebox"] #["DT_DOC3_hres"]#["DT"] #["DOC1_DT_Timebox"] #["DOC1_DT"]#["DOC1_DT_FULL"]#["DOC1_DT_Timebox"], "DOC1_DT_Occupancy", "DOC1_DT_Trigger"]
    
    available_subsystems = {"DT_DOC3_EFF","DT_DOC3_hres","DOC1_DT_Timebox","DOC1_DT_Occupancy","DOC1_DT_Trigger","DOC1_DT_FULL","DT_Online","DT_Online_Noise","DT_Online_ROS","DT_Online_Occupancy","DT_Online_Segments","DT_Online_Trigger","DT_Online_Trigger_PhiIn","DT_Online_Trigger_PhiOut"}    
    
    # End of setup variables - now we can start the code!
    ##############################################################################################################
    
    # lets clean the PDF path first
    pdf_folder = f'../runoffline/out/pdfs/'
    if os.path.exists(pdf_folder):
        shutil.rmtree(pdf_folder)

    with open('sample_info.json', 'r') as file:
        data = json.load(file)
    
    if(debug):
        data_runs = ["357754","357756"] #["357329"]#["355442"] #["355442"] #["360486"]#,"360486","360441","360400"]
    else:
        data_runs = data['runs'] #["357813","357814","357815","357898","356476","356489","356488"]
    bad_runs = data['bad_runs']
    
    # checking if the chosen class of histogram is inside the histogram_type list!
    is_subset = set(histogram_type).issubset(set(available_subsystems))
    if is_subset is False:
        print('ERROR: Chosen histogram type is not currently implemented!')
        exit() 

    # Looping over the trhesholds for the metastudies
    for metastudies_threshold in metastudies_thresholds:

        # Creating a directory with each histogram type to store the plots
        for hist_type in histogram_type:
            # Create the directory if it doesn't exist
            directory_path = './plots/' + str(hist_type)
            try:
                os.makedirs(directory_path, exist_ok=True)
                print(f"Directory '{directory_path}' created successfully or already exists.")
            except OSError as error:
                print(f"Error creating directory {directory_path}: {error}")

        # Now we make the loop over the thresholds!
        mean_number_true_positives = []
        mean_number_false_positives = []
        
        # Vectors to keep the values of Chi2 and MaxPull values
        values_thre_chi2, values_thre_MaxPull = [],[]
        
        #for threshold in vector_threshold:
        for nRefRuns,n_trial in zip(n_ref_runs, n_trials):
            
            # Loop over the number of trials, which is desirable for lower number of reference runs!
            for trial in range(n_trial):
            
                threshold = -1000
                results = []
                with ThreadPoolExecutor(max_workers=32) as executor:
                    
                    # Prepare a list of tasks for the progress bar
                    tasks = {executor.submit(process_run, hist_type, data_run, bad_runs, debug, nRefRuns, online,threshold): (hist_type, data_run) for hist_type in histogram_type for data_run in data_runs}
                    
                    # Initialize the progress bar with the total number of tasks
                    with tqdm(total=len(tasks), desc="Processing", unit="run") as pbar:
                        for future in as_completed(tasks):
                            hist_type, data_run = tasks[future]
                            try:
                                results.append(future.result())
                            except Exception as exc:
                                print(f'{hist_type} run {data_run} generated an exception: {exc}')
                            finally:
                                # Update the progress bar after each task is completed
                                pbar.update(1)
                
                for hist_type in histogram_type:
                    n_true_flagged_histos = []
                    n_false_flag_histos = []
                    
                    max_pull_true, max_chi2_true = [],[]
                    max_pull_false, max_chi2_false = [],[]
                    
                    max_pull_values_sorted,max_chi2_values_sorted = [],[]

                    filtered_results = [] # Initializing it as a empty array
                    filtered_results = [result for result in results if result[0] == hist_type]
                    
                    for _, data_run, num_pdfs, is_bad_run, max_pull_values, max_chi2_values in filtered_results:
                        
                        # Lets plot only the 10 highest pull plot per run!
                        max_pull_values_sorted = np.sort(max_pull_values)
                        max_pull_values_sorted = max_pull_values_sorted[int( len(max_pull_values_sorted) - 10 ):]

                        # The same with Chi2
                        max_chi2_values_sorted = np.sort(max_chi2_values)
                        max_chi2_values_sorted = max_chi2_values_sorted[int( len(max_chi2_values_sorted) - 10 ):]

                        if is_bad_run:
                            n_true_flagged_histos.append(num_pdfs)
                            max_pull_true = np.concatenate([max_pull_true, max_pull_values_sorted])
                            max_chi2_true = np.concatenate([max_chi2_true, max_chi2_values_sorted ])
                        else:
                            n_false_flag_histos.append(num_pdfs)
                            max_pull_false = np.concatenate([max_pull_false, max_pull_values_sorted])
                            max_chi2_false = np.concatenate([max_chi2_false, max_chi2_values_sorted])
                    # End of the loop over the filtered results - a.k.a. the results of the run offline script!
                    
                    for flag_type, data, max_pull, max_chi2 in [('False_flags', n_false_flag_histos, max_pull_false,max_chi2_false), ('True_flags', n_true_flagged_histos, max_pull_true,max_chi2_true)]:
                        
                        # Now we plot the Chi2 and MaxPull distributions and their thresholds
                        plt.figure()
                        hist_mean = np.nan_to_num(np.mean(max_pull))
                        hist_std  = np.nan_to_num(np.std(max_pull))

                        plt.hist(max_pull, bins=20, range=(0.0,hist_mean + 3.5*hist_std), edgecolor='black', label=f'Mean: {round(np.nan_to_num(np.mean(max_pull)),2)}')
                        
                        # Now, calculating the thresholds, such as 95% of the distributions is lower than that value
                        try:
                            sorted_max_pull = np.sort(max_pull)
                            max_pull_treshold = sorted_max_pull[int(metastudies_threshold*len(sorted_max_pull))]
                            values_thre_MaxPull.append( max_pull_treshold )
                        
                            # Creating a example plot to store the treshold in the plot
                            plt.plot([], [], ' ', label = f'threshold ({metastudies_threshold}): {max_pull_treshold}')
                            plt.plot([], [], ' ', label = f'Using {nRefRuns} Ref runs' )
                            plt.plot([], [], ' ', label=f'Number of histograms {len(sorted_max_pull)}')
                        except:
                            pass

                        plt.xlabel('Max pull')
                        plt.ylabel('Frequency')
                        plt.title(f'Max pull - {flag_type} - {hist_type}')
                        plt.legend(fontsize=10)
                        plt.savefig(f'./plots/{hist_type}/pull_{flag_type}_{hist_type}.png')  # Adjust path as needed
                        plt.close()

                        ###############################################
                        # Now the max chi2
                        hist_mean = np.nan_to_num(np.mean(max_chi2))
                        hist_std = np.nan_to_num(np.std(max_chi2))
                        plt.figure()

                        plt.hist(max_chi2, bins=20, range=(0.0,hist_mean + 3.5*hist_std), color = 'red',edgecolor='black', label=f'Mean: {round(np.nan_to_num(np.mean(max_chi2)),2)}')

                        # Now, calculating the thrsholds, such as 95% of the distributions is lower than that value
                        # it fails because the max_chi2 is empy sometimes, but why? -  because of the false and true flags ...  Since we dont have bad runs it always returns zero!
                        try:
                            sorted_max_chi2  = np.sort(max_chi2)
                            max_chi2_treshold = sorted_max_chi2[int(metastudies_threshold*len(sorted_max_chi2))]
                            values_thre_chi2.append(max_chi2_treshold)

                            # Creating a example plot to store the treshold in the plot
                            plt.plot([], [], ' ', label=f'threshold ({metastudies_threshold}): {max_chi2_treshold}')
                            plt.plot([], [], ' ', label=f'Using {nRefRuns} Ref runs')
                            plt.plot([], [], ' ', label=f'Number of histograms {len(sorted_max_chi2)}')
                        except:
                            pass

                        plt.xlabel('Max Chi2')
                        plt.ylabel('Frequency')
                        plt.title(f'Max Chi2 - {flag_type} - {hist_type}')
                        plt.legend(fontsize = 10)
                        plt.savefig(f'./plots/{hist_type}/chi2_{flag_type}_{hist_type}.png')  # Adjust path as needed
                        plt.close()     
                
                # Deleteing the vectors so they dont stack up!
                del max_chi2_false 
                del max_chi2_true
                del max_pull_false
                del max_pull_true
                
                # This below here are for the ROC curves, lets no use it for now!
                # Lets save the mean number of false and true positives
                #mean_number_true_positives.append(  np.nan_to_num(np.mean(n_true_flagged_histos)) )
                #mean_number_false_positives.append( np.nan_to_num(np.mean(n_false_flag_histos))  )
        
        # Now, lets print the thresholds and the nref runs in a .txt to further reading!
        # Writing data to the text file
        with open(f'./plots/{hist_type}/thresholds_{metastudies_threshold}.txt', 'w') as file:
            
            # Iterate over each pair of nref and ntrials
            for nref_val, ntrial_val in zip(n_ref_runs, n_trials):
                
                # Get the corresponding values for the current ntrial_val
                trial_values_CHI2    = values_thre_chi2[:ntrial_val]
                trial_values_maxpull = values_thre_MaxPull[:ntrial_val]
                
                # Print each pair of nref and value
                for value_maxpull, value_CHI2 in zip(trial_values_maxpull, trial_values_CHI2):
                    file.write(f'{nref_val}, {value_CHI2}, {value_maxpull}\n')
                    #file.write(f'nref, chi2, maxpull: {nref_val}, {value_CHI2}, {value_maxpull}\n')
                
                # Remove the used values from the list
                del values_thre_chi2[:ntrial_val]
                del values_thre_MaxPull[:ntrial_val]
        
        # Lets delet the vectors to make sure they dont stack up!
        del values_thre_chi2
        del values_thre_MaxPull
        del max_chi2_values
        del max_pull_values
        del max_chi2_treshold
        del sorted_max_chi2 
        del sorted_max_pull
        del max_pull_treshold
        del filtered_results
        del max_chi2
        del max_pull
        
    # Testing reading and making the fits plots!
    # Now lets ope the files and already perform the fits! We can save lots of time!
    dataframes = []
    for metastudies_threshold in metastudies_thresholds:
        for hist_type in histogram_type:
            file_path = f'./plots/{hist_type}/thresholds_{metastudies_threshold}.txt'
            dataframes.append(pd.read_csv(file_path, sep=',\s+', engine='python', header=None, names=['nref', 'chi2', 'maxpull']))
            #dataframes.append(pd.read_csv(file_path, sep=',\s+', engine='python', header=None, names=['nref', 'chi2', 'maxpull']))
    #print(dataframes) 
    plot_utils.fits_plotter(metastudies_thresholds, dataframes, histogram_type)          
        
if __name__ == '__main__':
    main()

