import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Data to plot
'''labels = ['AUC', 'Accuracy', 'F1-score', 'Precision', 'Recall', 'mAP']
n_labels = len(labels)

# Data for each model

mha_pe =     [0.9068, 0.7267, 0.8935, 0.8298 , 0.8875, 0.9308]
mha_r =     [0.9068, 0.7267, 0.8935, 0.8298 , 0.8875, 0.9308]
rmha_atte = [0.7569, 0.8567, 0.8176, 0.7086, 0.6804, 0.7664]
rmha_feat = [0.9232, 0.8350, 0.9030, 0.8559, 0.8455, 0.9022]
rmha =      [0.9394, 0.9091, 0.9293, 0.8901, 0.9183, 0.9763]'''

labels = ['AUC', 'Accuracy', 'F1-score', 'mAP']
n_labels = len(labels)

# Data for each model

mha_pe =     [0.9068, 0.7267, 0.8935, 0.9308]
mha_r =     [0.9068, 0.7267, 0.8935, 0.9308]
rmha_atte = [0.7569, 0.8567, 0.8176, 0.7664]
rmha_feat = [0.9232, 0.8350, 0.9030, 0.9022]
rmha =      [0.9394, 0.9091, 0.9293, 0.9763]
# Convert to polar coordinates
angles = [n / float(n_labels) * 2 * pi for n in range(n_labels)]
angles += angles[:1]  # Complete the loop

# Function to add data and plot each model
def add_model_data(values, label, color):
    values += values[:1]  # Complete the loop for each model
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=label, color=color)
    ##ax.fill(angles, values, color=color, alpha=0.25)

# Plot setup
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

# Set the background color to lime
ax.set_facecolor('azure')
##ax.set_facecolor('bisque')

# Plot each model
add_model_data(mha_pe, 'R-MHAGNN-pe', 'bisque')
add_model_data(mha_r, 'MHAGNN', 'tomato')
add_model_data(rmha_atte, 'R-MHAGNN-atte', 'cyan')
add_model_data(rmha_feat, 'R-MHAGNN-feat', 'sandybrown')
add_model_data(rmha, 'R-MHAGNN', 'blue')

# Labels and title
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=8)
ax.set_yticks([0.5, 1])
ax.set_yticklabels(['50%', '100%'], fontsize=8)
ax.yaxis.set_tick_params(labelsize=8)

# Add a legend, positioned on the right with more space between the plot and the legend
plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=9)  # Increased the horizontal distance

# Adjust layout to reduce left margin or make the plot tight
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # Adjust the left margin
plt.tight_layout()  # This will make the plot layout tighter

# Save the plot as a PNG file
plt.savefig('link_prediction_gat/results/radar_plot.png', format='png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
