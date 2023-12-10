import matplotlib.pyplot as plt
import pandas as pd

# This function plots similarity score vs mu averaged over runs
def plotting_mu_change(path_name,params):
        csv_file_path = path_name + "/result_stream.csv" # This might change, be wary
        
        result_df = pd.read_csv(csv_file_path)
        runs  = (result_df['run_no'].unique())
        df_grouped = result_df.groupby('mu').agg(['mean', 'std'])
        plt.figure(figsize=(7,4))

        for column in df_grouped.columns.levels[0][1:]:
            
            mean_values = df_grouped[column]['mean']
            std_values = df_grouped[column]['std']
            
            plt.plot(mean_values.index, mean_values, '-o',label=column)
            plt.fill_between(mean_values.index, mean_values - std_values, mean_values + std_values, alpha=0.2)

        plt.xlabel(r'Mixing Parameter: $\mu$')
        plt.ylabel('Element Centric Similarity')
        plt.legend(title="Algorithm", loc='upper right', bbox_to_anchor=(1.3, 0.8))
        plt.grid(True)
        plt.tight_layout()
        
        plt.title(rf'Runs: {len(runs)} | Nodes: {params["N"]} | $\tau$: {params["tau"]} | $<k>$: {params["k"]}')


        # Save the figure
        plt.savefig(f"{path_name}/experiment_plot.png",bbox_inches='tight')
        plt.close()
        return
    