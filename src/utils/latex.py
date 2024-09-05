import os
import sys
import json
import pandas
import shutil

if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))



def create_latex_table(results_folder, output_folder, methods, n_tasks, dataset):
    # Create a pandas dataframe
    columns = ['Method'] + ['Task {}'.format(i) for i in range(0, n_tasks)]
    df = pandas.DataFrame(columns=columns)
    # Create a folder for the figures
    if not os.path.exists(os.path.join(output_folder, 'figures')):
        os.makedirs(os.path.join(output_folder, 'figures'))
    # Iterate over methods
    for method in methods:
        method_name = method.replace('_', ' ')
        print('Processing method: {}'.format(method_name))
        # Copy figure to the output folder
        if method_name not in ['Oracle', 'Oracle-BN']:
            shutil.copyfile(os.path.join(results_folder, dataset, method, 'test_metrics.png'), 
                os.path.join(output_folder, 'figures', method + '.png'))
        # Load json results
        with open(os.path.join(results_folder, dataset, method, 'test_metrics.json')) as f:
            results = json.load(f)
        # Add row to the dataframe
        if method_name in ['Oracle', 'Oracle-BN']:
            df.loc[len(df.index)] = [method_name] + \
                [str(round(results['global']['mean'], 1)) for i in range(1, n_tasks + 1)]
        else:
            df.loc[len(df.index)] = [method_name] + \
                [str(round(results['global']['mean'][i-1], 1)) + '/' + str(round(results['local']['mean'][i-1], 1)) for i in range(1, n_tasks + 1)] 
    # Save dataframe in latex format
    df.to_latex(os.path.join(output_folder, 'results.tex'), index=False, column_format='l' + 'c' * n_tasks, escape=False)




if __name__ == "__main__":
    # Save results in latex format
    results_folder = '/ogr_cmu/output'
    n_tasks = 7
    dataset = sys.argv[1]
    methods = sys.argv[2].split(' ')

    # Create a folder for the results
    latex_folder = os.path.join(results_folder, dataset, 'LaTeX')
    if not os.path.exists(latex_folder):
        os.makedirs(latex_folder)
    
    # Create latex table
    create_latex_table(results_folder, latex_folder, methods, n_tasks, dataset)

