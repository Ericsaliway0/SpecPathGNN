import numpy as np
import matplotlib.pyplot as plt
from math import pi

# -----------------------
# Data
# -----------------------
labels = ['AUC', 'Accuracy', 'F1-score', 'mAP']
n_labels = len(labels)

mha_pe     = [0.9068, 0.7267, 0.8935, 0.9308]
mha_r      = [0.9068, 0.7267, 0.8935, 0.9308]
rmha_atte  = [0.7569, 0.8567, 0.8176, 0.7664]
rmha_feat  = [0.9232, 0.8350, 0.9030, 0.9022]
rmha       = [0.9394, 0.9091, 0.9293, 0.9763]

# -----------------------
# Polar coordinates
# -----------------------
angles = [n / float(n_labels) * 2 * pi for n in range(n_labels)]
angles += angles[:1]

def add_model_data(values, label, color):
    values = values + values[:1]
    ax.plot(angles, values, linewidth=2, label=label, color=color)

# -----------------------
# Plot
# -----------------------
fig, ax = plt.subplots(figsize=(8, 7.5), subplot_kw=dict(polar=True))
ax.set_facecolor('azure')

add_model_data(mha_pe,    'SpecPathGNN-pe',   'bisque')
add_model_data(mha_r,     'MHAGNN',         'tomato')
add_model_data(rmha_atte, 'SpecPathGNN-atte',  'cyan')
add_model_data(rmha_feat, 'SpecPathGNN-feat',  'sandybrown')
add_model_data(rmha,      'SpecPathGNN',       'blue')

# -----------------------
# Axis formatting
# -----------------------
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=14)

ax.set_yticks([0.5, 1.0])
ax.set_yticklabels(['50%', '100%'], fontsize=12)

# Move radial labels + protect visibility
ax.set_rlabel_position(22.5)
ax.set_axisbelow(True)

for label in ax.get_yticklabels():
    label.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.9))

# -----------------------
# CENTERED LEGEND (exact)
# -----------------------
ax.legend(
    loc="center",
    bbox_to_anchor=(0.5, 0.5),
    bbox_transform=ax.transAxes,
    fontsize=12,
    frameon=True,
    framealpha=0.95
)

# -----------------------
# Layout
# -----------------------
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

plt.savefig('radar_plot.png', dpi=300, bbox_inches='tight')
plt.show()
