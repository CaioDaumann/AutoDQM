import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Import tqdm for the progress bar
import shutil

def process_run(hist_type, data_run, bad_runs, reference_runs_str,debug, online):

    # Command to run AutoDQM
    if(online):
        data_ref_series = '000' + str(data_run)[:2] + 'xxxx'
        data_ref_sample = '000' + str(data_run)[:4] + 'xx'
        cmd = f"./run-offline.py Online {hist_type} {data_ref_series} {data_ref_sample} 370772 370293_370294 --ref_series 00035xxxx --ref_sample 0003551xx"
        if(debug):
            subprocess.run(cmd, shell=True, cwd="../../runoffline/")
        else:
            subprocess.run(cmd, stdout=subprocess.DEVNULL,shell=True, cwd="../../runoffline/")
    else:
        # For offline things change a bit ...
        data_ref_series = "Run2022"
        data_ref_sample = "SingleMuon" 
        cmd = f"./run-offline.py Offline DT Run2022 SingleMuon {data_run} 355135_355134"
        if(debug):
            subprocess.run(cmd, shell=True, cwd="../../runoffline/")
        else:
            subprocess.run(cmd, stdout=subprocess.DEVNULL,shell=True, cwd="../../runoffline/")

    pdf_folder = f'../../runoffline/out/pdfs/{hist_type}/{data_run}'
    
    pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith('.pdf')]
    num_pdfs = len(pdf_files)
    
    txt_files = [file for file in os.listdir(pdf_folder) if file.endswith('.txt')]
    max_pull_values = []
    for txt_file in txt_files:
        with open(pdf_folder + '/' + txt_file, 'r') as file:
            # Read the content of the file
            content = file.read()
            max_pull_values.append(abs(float(content)))        

    # Now we exclude the pdfs and txt so we dont double count next run!
    for pdf_file in pdf_files:
        os.remove(os.path.join(pdf_folder, pdf_file))
    
    for txt_file in txt_files:
        os.remove(pdf_folder + '/' + txt_file)

    is_bad_run = data_run in bad_runs

    return hist_type, data_run, num_pdfs, is_bad_run, max_pull_values

def main():
    
    print('Remenber to set the enviroment variable manually!')
    
    # Setup variables
    debug = True
    online = True
    
    # lets clean the PDF path first
    pdf_folder = f'../../runoffline/out/pdfs/'
    if os.path.exists(pdf_folder):
        shutil.rmtree(pdf_folder)

    with open('sample_info.json', 'r') as file:
        data = json.load(file)
    
    if(debug):
        data_runs = ["370144"]
    else:
        data_runs = data['runs'] #["357813","357814","357815","357898","356476","356489","356488"]
    bad_runs = data['bad_runs']
    histogram_type = ["DT"]#["DOC1_DT"]#Timebox", "DOC1_DT_Occupancy", "DOC1_DT_Trigger"]
    reference_runs = [355872, 355892, 355912, 355913, 355921, 355988, 356005, 356043]
    reference_runs_str = '_'.join(map(str, reference_runs))
    
    results = []
    with ThreadPoolExecutor(max_workers=60) as executor:
        # Prepare a list of tasks for the progress bar
        tasks = {executor.submit(process_run, hist_type, data_run, bad_runs, reference_runs_str,debug, online): (hist_type, data_run) for hist_type in histogram_type for data_run in data_runs}
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
        
        max_pull_true = []
        max_pull_false = []

        filtered_results = [result for result in results if result[0] == hist_type]
        
        for _, data_run, num_pdfs, is_bad_run, max_pull_values in filtered_results:
            if is_bad_run:
                n_true_flagged_histos.append(num_pdfs)
                max_pull_true = np.concatenate([max_pull_true, max_pull_values])
            else:
                n_false_flag_histos.append(num_pdfs)
                max_pull_false = np.concatenate([max_pull_false, max_pull_values])
        
        for flag_type, data, max_pull in [('False_flags', n_false_flag_histos, max_pull_false), ('True_flags', n_true_flagged_histos, max_pull_true)]:
            plt.figure()
            plt.hist(data, bins=20, range=(0,100), edgecolor='black', label=f'Mean: {np.nan_to_num(np.mean(data))}')
            plt.xlabel('Number of flagged histograms')
            plt.ylabel('Frequency')
            plt.title(f'{flag_type} {hist_type}')
            plt.legend()
            plt.savefig(f'./plots/{hist_type}/{flag_type}_{hist_type}.png')  # Adjust path as needed
            plt.close()

            plt.figure()
            plt.hist(max_pull, bins=25, range=(0,50), edgecolor='black', label=f'Mean: {np.nan_to_num(np.mean(max_pull))}')
            plt.xlabel('Max pull')
            plt.ylabel('Frequency')
            plt.title(f'{flag_type} {hist_type}')
            plt.legend()
            plt.savefig(f'./plots/{hist_type}/pull_{flag_type}_{hist_type}.png')  # Adjust path as needed
            plt.close()
            

if __name__ == '__main__':
    main()
