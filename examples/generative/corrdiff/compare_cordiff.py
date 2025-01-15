import os
import json
import logging
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from matplotlib.backends.backend_pdf import PdfPages
import pdb

# Specify numerical experiments to be compared
experiment_dir = 'outputs/generation_scores/default/'
model_folders = [ 
    'patched_diffusion/1216822'
    ]

data_folders = [os.path.join(experiment_dir, folder) for folder in model_folders]

# Constants for figure sizes and layouts
FIG_WIDTH = 20
FIG_HEIGHT = 11
SUBPLOT_ADJUST = dict(left=0.06, right=0.94, top=0.90, bottom=0.08, hspace=0.3, wspace=0.2)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Function definitions

def load_json_safely(file_path):
    """Safely load a JSON file, handling potential errors."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning(f"Failed to decode JSON from {file_path}. File may be empty or improperly formatted.")
        return None
    except Exception as e:
        logger.warning(f"Error loading {file_path}: {str(e)}")
        return None

def read_scores(file_path):
    """Read scores from a netCDF file and convert to a structured dictionary."""
    try:
        with xr.open_dataset(file_path) as dataset:
            df = dataset.to_dataframe().reset_index()
            rename_dict = {
                'refc': 'maximum_radar_reflectivity',
                '2t': 'temperature_2m',
                '10u': 'eastward_wind_10m',
                '10v': 'northward_wind_10m'
            }
            df = df.rename(columns=rename_dict)
            df['time'] = df['time'].dt.strftime("%Y-%m-%dT%H:%M:%S")
            grouped = df.groupby('metric')
            all_data = {name: group.to_dict('list') for name, group in grouped}
            return all_data
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return None

def process_folder(data_folder):
    """Process a single data folder, loading various data files."""
    # file_path = os.path.join(data_folder, "scores.nc")
    file_path = os.path.join(data_folder, "score.nc")
    config_path = os.path.join(data_folder, "config.json")
    spectra_path = os.path.join(data_folder, "spectra", "spectra.json")
    distributions_path = os.path.join(data_folder, "spectra", "distributions.json")
    # rank_hist_path = os.path.join(data_folder, "dispersion", "rank_hist.json")
    fid_path = os.path.join(data_folder, "fid", "fid.json")

    config = load_json_safely(config_path) if os.path.exists(config_path) else None
    spectra = load_json_safely(spectra_path) if os.path.exists(spectra_path) else None
    distributions = load_json_safely(distributions_path) if os.path.exists(distributions_path) else None
    fid = load_json_safely(fid_path) if os.path.exists(fid_path) else None
    # rank_hist = load_json_safely(rank_hist_path) if os.path.exists(rank_hist_path) else None
    all_data = read_scores(file_path) if os.path.exists(file_path) else None

    return {
        'file_path': file_path,
        'config': config,
        'spectra': spectra,
        'distributions': distributions,
        'fid': fid,
        # 'rank_hist': rank_hist,
        'all_data': all_data
    }

def process_multiple_folders(data_folders):
    """Process multiple data folders and return combined results."""
    results = {}
    for folder in data_folders:
        logger.info(f"Processing folder: {folder}")
        results[folder] = process_folder(folder)
    return results

def plot_metric_comparison(all_results, model_folders, metric_name, pdf):
    """Generate a plot comparing a specific metric across different models."""
    variables = ['temperature_2m', 'eastward_wind_10m', 'northward_wind_10m', 'maximum_radar_reflectivity']
    metric_full_names = {
        'crps': 'Continuous Ranked Probability Score',
        'mae': 'Mean Absolute Error',
        'rmse': 'Root Mean Square Error'
    }
    
    fig, axs = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.suptitle(f'{metric_full_names[metric_name]} Comparison Across Models', fontsize=24, fontweight='bold')

    labels = [os.path.dirname(folder) for folder in model_folders]
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    def style_boxplot(bp, colors):
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=2)
        for patch, color in zip(bp['boxes'], colors):
            patch.set(facecolor=color, edgecolor='black', linewidth=2)

    for idx, variable in enumerate(variables):
        ax = axs[idx // 2, idx % 2]
        metric_data = [model_data['all_data'][metric_name][variable] 
                       for model_data in all_results.values() 
                       if model_data['all_data'] is not None and metric_name in model_data['all_data']]
        
        if metric_data:
            bp = ax.boxplot(metric_data, vert=False, patch_artist=True, widths=0.6,
                            flierprops=dict(marker='o', markersize=5, linestyle='none'))
            style_boxplot(bp, colors[:len(metric_data)])

            ax.set_xlabel(f'{metric_full_names[metric_name]} (lower is better)', fontsize=14, fontweight='bold')
            ax.set_title(f'{metric_full_names[metric_name]} for {variable}', fontsize=16, fontweight='bold')
            ax.set_xlim(0, max(1.5, max(max(data) for data in metric_data) * 1.1))
            ax.invert_yaxis()
            ax.set_yticks([])
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            ax.tick_params(axis='x', labelsize=12)

            for i, median in enumerate(bp['medians']):
                ax.text(median.get_xdata()[0], i+1, f'{median.get_xdata()[0]:.3f}', 
                        va='center', ha='left', fontweight='bold', fontsize=12)

            ax.set_aspect(0.5 * ax.get_xlim()[1] / (len(metric_data) + 1))
        else:
            ax.text(0.5, 0.5, f'No data available for {variable}', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)

    plt.tight_layout()
    fig.subplots_adjust(**SUBPLOT_ADJUST)

    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black') for color in colors]
    
    # Place legend at the center of the figure
    legend = fig.legend(legend_elements, labels, loc='center', title='Models', 
                        title_fontsize='large', fontsize=12, bbox_to_anchor=(0.5, 0.5),
                        bbox_transform=fig.transFigure)
    
    # Set semi-transparent background for legend
    legend.get_frame().set_alpha(0.8)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    # Add metric explanation
    fig.text(0.5, 0.01, f'{metric_full_names[metric_name]} measures the accuracy of predictions.\nLower values indicate better performance.', 
             ha='center', va='center', fontsize=12, fontweight='bold')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def plot_multi_model_spectra(all_results, model_folders, fields, pdf):
    """Plot spectra for multiple models and fields, including ratios to truth."""
    n_models = len(model_folders)
    n_fields = len(fields)
    
    fig, axs = plt.subplots(n_fields, 2, figsize=(FIG_WIDTH, FIG_HEIGHT * n_fields / 2), squeeze=False)
    fig.suptitle('Spectra Comparison Across Models', fontsize=24, fontweight='bold')
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    model_names = [os.path.dirname(folder) for folder in model_folders]

    for field_idx, field in enumerate(fields):
        ax1, ax2 = axs[field_idx]
        
        for model_idx, (model, results) in enumerate(all_results.items()):
            spectra = results.get('spectra', {})
            if not spectra:
                print(f"Warning: No spectra data found for {model}")
                continue
            
            data = spectra.get(field, {})
            if 'prediction' in data:
                ax1.loglog(*data['prediction'], label=model_names[model_idx], color=colors[model_idx])
            else:
                print(f"Warning: Prediction data not found for {field} in {model}")
        
        truth_data = next((results['spectra'].get(field, {}).get('truth') 
                           for results in all_results.values() 
                           if results.get('spectra') and 'truth' in results['spectra'].get(field, {})), None)
        
        if truth_data:
            ax1.loglog(*truth_data, label='Truth', color='black', linewidth=2)
        else:
            print(f"Warning: Truth data not found for {field}")
        
        ax1.set(xlabel='Frequency', ylabel='Power', title=f'{field.capitalize()} Spectra')
        ax1.set_ylim(bottom=1e-2)
        ax1.legend()
        ax1.grid(True, which="both", ls="-", alpha=0.2)

        if truth_data:
            truth_freq, truth_power = np.array(truth_data[0]), np.array(truth_data[1])
            for model_idx, (model, results) in enumerate(all_results.items()):
                spectra = results.get('spectra', {})
                if not spectra:
                    continue
                
                model_data = spectra.get(field, {})
                if 'prediction' in model_data:
                    pred_freq, pred_power = np.array(model_data['prediction'][0]), np.array(model_data['prediction'][1])
                    if not np.array_equal(pred_freq, truth_freq):
                        pred_power = np.interp(truth_freq, pred_freq, pred_power)
                    ratio = pred_power / truth_power
                    ax2.semilogx(truth_freq, ratio, label=model_names[model_idx], color=colors[model_idx])
                else:
                    print(f"Warning: Prediction data not found for {field} in {model}")
            
            ax2.set(xlabel='Frequency', ylabel='Power Ratio', title=f'{field.capitalize()} Spectra Ratio to Truth')
            ax2.axhline(y=1, color='r', linestyle='--', label='Truth reference')
            ax2.set_ylim(0, 2)
            ax2.legend()
            ax2.grid(True, which="both", ls="-", alpha=0.2)
        else:
            ax2.text(0.5, 0.5, "Truth data not available", ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    fig.subplots_adjust(**SUBPLOT_ADJUST)
    pdf.savefig(fig)
    plt.close(fig)

def plot_multi_model_distributions(all_results, model_folders, fields, pdf):
    """Plot probability distributions for multiple models and fields, including ratios to truth."""
    n_models = len(model_folders)
    n_fields = len(fields)
    
    fig, axs = plt.subplots(n_fields, 2, figsize=(FIG_WIDTH, FIG_HEIGHT * n_fields / 2), squeeze=False)
    fig.suptitle('Probability Distribution Comparison Across Models', fontsize=24, fontweight='bold')
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    model_names = [os.path.dirname(folder) for folder in model_folders]

    field_mapping = {
        'velocities': ['wind_speed', 'velocities'],
        'temperature': ['temperature', 'temperature_2m'],
        'reflectivity': ['reflectivity', 'maximum_radar_reflectivity']
    }

    for field_idx, field in enumerate(fields):
        ax1, ax2 = axs[field_idx]
        
        has_data = False
        possible_fields = field_mapping.get(field, [field])
        
        for model_idx, (model, results) in enumerate(all_results.items()):
            distributions = results.get('distributions', {})
            if not distributions:
                print(f"Warning: No distribution data found for {model}")
                continue
            
            data = next((distributions[possible_field] for possible_field in possible_fields if possible_field in distributions), None)
            
            if data and 'prediction' in data:
                x, y = data['prediction']
                ax1.semilogy(x, y, label=f"{model_names[model_idx]} (Pred)", color=colors[model_idx])
                has_data = True
            else:
                print(f"Warning: Prediction data not found for {field} in {model}")
        
        truth_data = next((distributions[possible_field]['truth']
                           for results in all_results.values()
                           for possible_field in possible_fields
                           if results.get('distributions') and possible_field in results['distributions']
                           and 'truth' in results['distributions'][possible_field]),
                          None)
        
        if truth_data:
            x, y = truth_data
            ax1.semilogy(x, y, label='Truth', color='black', linewidth=2)
            has_data = True
        else:
            print(f"Warning: Truth data not found for {field}")
        
        if has_data:
            ax1.set(xlabel='Value', ylabel='Probability Density (log scale)', 
                    title=f'{field.capitalize()} Probability Distributions')
            ax1.grid(True, which="both", ls="-", alpha=0.2)
            ax1.legend()

            if truth_data:
                truth_x, truth_y = np.array(truth_data)
                truth_interp = interpolate.interp1d(truth_x, truth_y, bounds_error=False, fill_value='extrapolate')
                
                all_ratios = []
                for model_idx, (model, results) in enumerate(all_results.items()):
                    distributions = results.get('distributions', {})
                    if not distributions:
                        continue
                    
                    data = next((distributions[possible_field] for possible_field in possible_fields if possible_field in distributions), None)
                    
                    if data and 'prediction' in data:
                        pred_x, pred_y = np.array(data['prediction'])
                        
                        min_x = max(min(truth_x), min(pred_x))
                        max_x = min(max(truth_x), max(pred_x))
                        x_interp = np.linspace(min_x, max_x, 1000)
                        
                        pred_interp = interpolate.interp1d(pred_x, pred_y, bounds_error=False, fill_value='extrapolate')
                        
                        truth_y_interp = truth_interp(x_interp)
                        pred_y_interp = pred_interp(x_interp)
                        
                        ratio = np.where((truth_y_interp != 0) & (pred_y_interp != 0),
                                         pred_y_interp / truth_y_interp, np.nan)
                        
                        ax2.semilogy(x_interp, ratio, label=model_names[model_idx], color=colors[model_idx])
                        all_ratios.append(ratio)
                    else:
                        print(f"Warning: Prediction data not found for {field} in {model}")
                
                ax2.axhline(y=1, color='r', linestyle='--', label='Equal ratio')
                ax2.set(xlabel='Value', ylabel='Ratio (Prediction / Truth) (log scale)',
                        title=f'{field.capitalize()} Distribution Ratio (Prediction / Truth)')
                ax2.grid(True, which="both", ls="-", alpha=0.2)
                ax2.legend()
                
                if all_ratios:
                    valid_ratios = np.concatenate([r[np.isfinite(r) & (r > 0)] for r in all_ratios])
                    if len(valid_ratios) > 0:
                        ylim = np.percentile(valid_ratios, [5, 95])
                        ax2.set_ylim(max(ylim[0] * 0.1, min(valid_ratios)), min(ylim[1] * 10, max(valid_ratios)))
                    else:
                        print(f"Warning: No valid ratio values for {field}")
                        ax2.set_ylim(0.1, 10)
                else:
                    print(f"Warning: No ratio data available for {field}")
                    ax2.set_ylim(0.1, 10)
            else:
                ax2.text(0.5, 0.5, "Truth data not available", ha='center', va='center', transform=ax2.transAxes)
        else:
            ax1.text(0.5, 0.5, f"No data available for {field}", ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, f"No data available for {field}", ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    fig.subplots_adjust(**SUBPLOT_ADJUST)
    pdf.savefig(fig)
    plt.close(fig)

def main():
    """Main function to process data and generate the PDF report."""
    # Process data
    all_results = process_multiple_folders(data_folders)

    # Generate PDF report
    with PdfPages('corrdiff_comparison_report_10m.pdf') as pdf:
        # Metric comparisons
        metrics = ['crps', 'mae', 'rmse']
        for metric in metrics:
            plot_metric_comparison(all_results, model_folders, metric, pdf)

        # Spectra plots
        fields = ['velocities', 'temperature', 'reflectivity']
        plot_multi_model_spectra(all_results, model_folders, fields, pdf)

        # Distribution plots
        plot_multi_model_distributions(all_results, model_folders, fields, pdf)

    print("PDF report generated: corrdiff_us_report.pdf")

if __name__ == "__main__":
    main()