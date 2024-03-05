import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Import tqdm for the progress bar
import shutil

def process_run(hist_type, data_run, bad_runs, reference_runs_str,debug, online,threshold=10):

    # Command to run AutoDQM
    if(online):
        data_ref_series = '000' + str(data_run)[:2] + 'xxxx'
        data_ref_sample = '000' + str(data_run)[:4] + 'xx'
        cmd = f"./run-offline.py Online {hist_type} {data_ref_series} {data_ref_sample} {data_run} 355872_361193_360892_359814_359812_361110_361994_359764 --ref_series 00035xxxx --ref_sample 0003551xx --threshold {threshold}"
        if(debug):
            subprocess.run(cmd, shell=True, cwd="../runoffline/")
        else:
            subprocess.run(cmd, stdout=subprocess.DEVNULL,shell=True, cwd="../runoffline/")
    else:
        # For offline things change a bit ...
        data_ref_series = "Run2022"
        data_ref_sample = "SingleMuon" 
        cmd = f"./run-offline.py Offline DT Run2022 SingleMuon {data_run} 355135_355134"
        if(debug):
            subprocess.run(cmd, shell=True, cwd="../runoffline/")
        else:
            subprocess.run(cmd, stdout=subprocess.DEVNULL,shell=True, cwd="../runoffline/")
    #print(cmd)
    pdf_folder = f'../runoffline/out/pdfs/{hist_type}/{data_run}'
    
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
    debug = False
    online = True
    
    # lets clean the PDF path first
    pdf_folder = f'../runoffline/out/pdfs/'
    # Uncomment this!
    if os.path.exists(pdf_folder):
        shutil.rmtree(pdf_folder)

    with open('sample_info.json', 'r') as file:
        data = json.load(file)
    
    if(debug):
        data_runs = ["355872"]#,"360486","360441","360400"]
    else:
        data_runs = data['runs'] #["357813","357814","357815","357898","356476","356489","356488"]
    bad_runs = data['bad_runs']
    histogram_type = ["DOC1_DT"] #["DOC1_DT_Timebox"] #["DOC1_DT"]#["DOC1_DT_FULL"]#["DOC1_DT_Timebox"], "DOC1_DT_Occupancy", "DOC1_DT_Trigger"]
    reference_runs = [355872, 355892, 355912, 355913, 355921, 355988, 356005, 356043]
    reference_runs_str = '_'.join(map(str, reference_runs))
    
    # Creating a directory with each histogram type to store the plots
    for hist_type in histogram_type:
        # Create the directory if it doesn't exist
        directory_path = './plots/' + str(hist_type)
        try:
            os.makedirs(directory_path, exist_ok=True)
            print(f"Directory '{directory_path}' created successfully or already exists.")
        except OSError as error:
            print(f"Error creating directory {directory_path}: {error}")

    # Now we make the loop over the tresholds!
    mean_number_true_positives = []
    mean_number_false_positives = []
    vector_threshold = [4,10,16]
    for threshold in vector_threshold:

        results = []
        with ThreadPoolExecutor(max_workers=600) as executor:
            # Prepare a list of tasks for the progress bar
            tasks = {executor.submit(process_run, hist_type, data_run, bad_runs, reference_runs_str,debug, online,threshold): (hist_type, data_run) for hist_type in histogram_type for data_run in data_runs}
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
                
                # Lets plot only the 10 higest pull plot per run!
                max_pull_values = np.sort(max_pull_values)
                max_pull_values = max_pull_values[int( len(max_pull_values) - 15 ):]

                if is_bad_run:
                    n_true_flagged_histos.append(num_pdfs)
                    max_pull_true = np.concatenate([max_pull_true, max_pull_values])
                else:
                    n_false_flag_histos.append(num_pdfs)
                    max_pull_false = np.concatenate([max_pull_false, max_pull_values])
            
            for flag_type, data, max_pull in [('False_flags', n_false_flag_histos, max_pull_false), ('True_flags', n_true_flagged_histos, max_pull_true)]:
                plt.figure()
                plt.hist(data, bins=13, range=(-0.5,12.5), edgecolor='black', label=f'Mean: {np.nan_to_num(np.mean(data))}')
                plt.xlabel('Number of flagged histograms')
                plt.ylabel('Frequency')
                plt.title(f'{flag_type} {hist_type}')
                plt.legend()
                plt.savefig(f'./plots/{hist_type}/{flag_type}_{hist_type}.png')  # Adjust path as needed
                plt.close()

                plt.figure()
                if "Timebox" in hist_type:
                    plt.hist(max_pull, bins=25, range=(0.0,0.45), edgecolor='black', label=f'Mean: {np.nan_to_num(np.mean(max_pull))}')
                else:
                    plt.hist(max_pull, bins=25, range=(-0.50,25.5), edgecolor='black', label=f'Mean: {np.nan_to_num(np.mean(max_pull))}')
                plt.xlabel('Max pull')
                plt.ylabel('Frequency')
                plt.title(f'Max pull - {flag_type} - {hist_type}')
                plt.legend()
                plt.savefig(f'./plots/{hist_type}/pull_{flag_type}_{hist_type}.png')  # Adjust path as needed
                plt.close()
            
        # Lets save the mean number of false and true positives
        mean_number_true_positives.append(  np.nan_to_num(np.mean(n_true_flagged_histos)) )
        mean_number_false_positives.append( np.nan_to_num(np.mean(n_false_flag_histos))  )

    # Plotting the scatter plot
    plt.plot(mean_number_false_positives, mean_number_true_positives, '-o', label = 'Beta-binomial')

    # Adding labels and title
    plt.xlabel('Mean number of histogram flags per good run')
    plt.ylabel('Mean number of histogram flags per bad run')
    plt.title('ROC curve')

    plt.plot([np.min(mean_number_false_positives), np.max(mean_number_false_positives)], [np.min(mean_number_false_positives), np.max(mean_number_false_positives)], color='black', lw=2, linestyle='--', label='Random guess')
    plt.legend(fontsize=12, loc='lower right')

    plt.savefig(f'./plots/{hist_type}/ROC_curve.png')

if __name__ == '__main__':
    main()
