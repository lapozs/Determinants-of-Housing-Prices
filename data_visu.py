"""
@author: Pozsgai Emil Csanád
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the new Excel file
file_path = 'Excel_base_FINAL-c1.xlsx'
data = pd.read_excel(file_path, header=0)

# Separate metadata and time series data
metadata = data.iloc[:, :4]  # First four columns for metadata
time_series_data = data.iloc[:, 4:]  # Time series data starts from the fifth column
time_series_data.index = metadata.iloc[:, 1]  # Set variable names as index

# Shorten titles for better visualization
def shorten_title(title, max_length=50):
    return title if len(title) <= max_length else title[:max_length] + '...'

shortened_titles = [shorten_title(name) for name in time_series_data.index]

# Plot each time series in a grid layout
num_plots = len(time_series_data.index)
cols = 5  # Number of columns
rows = (num_plots // cols) + (num_plots % cols > 0)  # Calculate number of rows needed

# Define a function to identify outliers using the IQR method
def identify_outliers(series):
    Q1 = np.percentile(series.dropna(), 25)
    Q3 = np.percentile(series.dropna(), 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers

# Assign colors based on categories (from the first column of metadata)
categories = metadata.iloc[:, 0].unique()
category_colors = {category: plt.cm.tab10(i+5) for i, category in enumerate(categories)} #plt.cm.tab20(i)
metadata['Color'] = metadata.iloc[:, 0].map(category_colors)

# Plot each time series with outliers and colored by category
fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows), sharex=False)
axes = axes.flatten()  # Flatten to 1D for easier indexing

for i, (var_name, short_title) in enumerate(zip(time_series_data.index, shortened_titles)):
    series = time_series_data.loc[var_name]
    category = metadata.loc[metadata.iloc[:, 1] == var_name, 'Color'].values[0]  # Get category color
    outliers = identify_outliers(series)
    
    # Plot time series
    axes[i].plot(series.index, series, color=category, marker='o', linestyle='-', label='Idősor')
    # Highlight outliers
    axes[i].scatter(outliers.index, outliers.values, marker='x', color='red', zorder=5, label='Kiugró értékek')
    axes[i].set_title(short_title, fontsize=10)
    axes[i].set_xticks(range(len(series.index)))  # Set ticks to include all years
    axes[i].set_xticklabels(series.index, rotation=45, fontsize=8)
    axes[i].tick_params(axis='y', labelsize=8)
    
    # Add legend to each subplot
    axes[i].legend(fontsize=8, loc='upper left')

# Turn off unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Main title and layout adjustment
fig.suptitle('Változók idősora (Kiugró értékek és kategóriák színezve)', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save and display the updated plot
output_path_outliers = 'Time_Series_Visualization.png'
plt.savefig(output_path_outliers, dpi=300)
plt.show()

output_path_outliers