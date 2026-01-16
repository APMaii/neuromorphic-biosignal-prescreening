
'''
This script performs exploratory data analysis (EDA) on the PSA biomarker dataset.

What it does:
1. Loads data: Reads the generated PSA dataset from CSV containing concentrations,
   frequencies, ratios, and risk level classifications
   
2. Visualizes concentration space: Creates scatter plots showing Free PSA vs Total PSA
   in concentration space, with risk level color-coding and ratio boundary lines
   
3. Visualizes frequency space: Displays the relationship between Free and Total PSA
   in the frequency domain (converted from concentrations)
   
4. Analyzes distributions: Generates histograms for concentrations, frequencies, and
   Free/Total PSA ratios to understand data distributions
   
5. Validates conversion formula: Creates log-log plots to verify the concentration-to-frequency
   conversion relationship follows the expected power law
   
6. Risk level analysis: Produces comprehensive multi-panel visualizations including:
   - Sample counts per risk level
   - Ratio distributions by risk level (box plots)
   - Concentration distributions by risk level (box plots)
   
All visualizations use professional styling with high-resolution output suitable for
publication and academic presentations.
'''


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# Set professional style parameters
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')
        mpl.rcParams['axes.grid'] = True
        mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 14




# Professional color palette - sophisticated and cohesive
# Risk levels: High Risk -> Low Risk (muted, professional, academic colors)
RISK_COLORS = {
    "High Risk": "#8B4A6B",           # Muted burgundy/wine
    "Moderately High Risk": "#B87333",  # Muted terracotta/rust
    "Intermediate": "#5A7D5A",         # Muted sage green
    "Low Risk": "#4A6FA5"              # Muted slate blue
}

# Professional palette for histograms - clearly distinguishable colors
HIST_COLORS = ["#0EA5E9", "#EF4444"]  # Sky blue and distinct red (clearly different)

# Reference line colors (subtle and professional)
REF_LINE_COLORS = {
    10: "#B85450",   # Muted red
    15: "#D4A574",   # Muted amber
    25: "#6B8E7E"    # Muted teal
}





# Ask the user to enter the path to the CSV file
path =input('Enter the path to the CSV file (Free_Total_PSA_frequency.csv):')
df = pd.read_csv(path)


# =========================
# 1. Free PSA vs Total PSA (Concentration)
# =========================
plt.figure(figsize=(10, 7))
for level in df["Risk_Level"].unique():
    sub = df[df["Risk_Level"] == level]
    color = RISK_COLORS.get(level, "#808080")
    plt.scatter(sub["Free_PSA_nM"], sub["Total_PSA_nM"], 
                label=level, s=25, alpha=0.7, color=color, edgecolors='white', linewidth=0.5)

# Add diagonal line (free_psa = total_psa) to show constraint
max_val = max(df["Total_PSA_nM"].max(), df["Free_PSA_nM"].max())
plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, linewidth=1.5, label='Free = Total (boundary)')

# Ratio boundaries (approximate)
# For visualization: lines showing ratio boundaries
x_line = np.linspace(0.01, max_val, 100)
plt.plot(x_line, x_line / 0.10, color=REF_LINE_COLORS[10], linestyle='--', 
         alpha=0.4, linewidth=1.5, label='10% ratio')
plt.plot(x_line, x_line / 0.15, color=REF_LINE_COLORS[15], linestyle='--', 
         alpha=0.4, linewidth=1.5, label='15% ratio')
plt.plot(x_line, x_line / 0.25, color=REF_LINE_COLORS[25], linestyle='--', 
         alpha=0.4, linewidth=1.5, label='25% ratio')

plt.xlabel("Free PSA (nM)", fontsize=26, fontweight='bold', labelpad=15)
plt.ylabel("Total PSA (nM)", fontsize=26, fontweight='bold', labelpad=15)

#plt.title("Free PSA vs Total PSA Concentration Space", fontweight='bold', pad=15)
plt.tick_params(axis='both', which='major',  labelsize=14,width=1.5, length=6)

plt.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
plt.xlim(0,1)
plt.ylim(0,1)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()




plt.figure(figsize=(10, 7))
for level in df["Risk_Level"].unique():
    sub = df[df["Risk_Level"] == level]
    color = RISK_COLORS.get(level, "#808080")
    plt.scatter(sub["Free_PSA_nM"], sub["Total_PSA_nM"], 
                label=level, s=25, alpha=0.7, color=color, edgecolors='white', linewidth=0.5)

# Add diagonal line (free_psa = total_psa) to show constraint
max_val = max(df["Total_PSA_nM"].max(), df["Free_PSA_nM"].max())
plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, linewidth=1.5, label='Free = Total (boundary)')

# Ratio boundaries (approximate)
# For visualization: lines showing ratio boundaries
x_line = np.linspace(0.01, max_val, 100)
plt.plot(x_line, x_line / 0.10, color=REF_LINE_COLORS[10], linestyle='--', 
         alpha=0.4, linewidth=1.5, label='10% ratio')
plt.plot(x_line, x_line / 0.15, color=REF_LINE_COLORS[15], linestyle='--', 
         alpha=0.4, linewidth=1.5, label='15% ratio')
plt.plot(x_line, x_line / 0.25, color=REF_LINE_COLORS[25], linestyle='--', 
         alpha=0.4, linewidth=1.5, label='25% ratio')

plt.xlabel("Free PSA (nM)", fontweight='medium')
plt.ylabel("Total PSA (nM)", fontweight='medium')
#plt.title("Free PSA vs Total PSA Concentration Space", fontweight='bold', pad=15)
plt.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
plt.xlim(0,8)
plt.ylim(0,8)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()






# =========================
# 2. Free PSA vs Total PSA (Frequency)
# =========================
plt.figure(figsize=(10, 7))
for level in df["Risk_Level"].unique():
    sub = df[df["Risk_Level"] == level]
    color = RISK_COLORS.get(level, "#808080")
    plt.scatter(
        sub["Free_PSA_freq_Hz"],
        sub["Total_PSA_freq_Hz"],
        label=level,
        s=25,
        alpha=0.7,
        color=color,
        edgecolors='white',
        linewidth=0.5
    )

plt.xlabel("Free PSA Frequency (Hz)", fontweight='medium')
plt.ylabel("Total PSA Frequency (Hz)", fontweight='medium')
#plt.title("Free PSA vs Total PSA Frequency Space", fontweight='bold', pad=15)
plt.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()





# =========================
# 3. Histograms (Concentration and Frequency)
# =========================
plt.figure(figsize=(10, 6))
plt.hist(df["Free_PSA_nM"], bins=30, alpha=0.75, label='Free PSA', 
         color=HIST_COLORS[0], edgecolor='white', linewidth=1.2)
plt.hist(df["Total_PSA_nM"], bins=30, alpha=0.75, label='Total PSA', 
         color=HIST_COLORS[1], edgecolor='white', linewidth=1.2)
plt.xlabel("PSA Concentration (nM)", fontweight='medium')
plt.ylabel("Count", fontweight='medium')
plt.title("PSA Concentration Distribution", fontweight='bold', pad=15)
plt.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df["Free_PSA_freq_Hz"], bins=30, alpha=0.75, label='Free PSA', 
         color=HIST_COLORS[0], edgecolor='white', linewidth=1.2)
plt.hist(df["Total_PSA_freq_Hz"], bins=30, alpha=0.75, label='Total PSA', 
         color=HIST_COLORS[1], edgecolor='white', linewidth=1.2)
plt.xlabel("PSA Frequency (Hz)", fontweight='medium')
plt.ylabel("Count", fontweight='medium')
plt.title("PSA Frequency Distribution", fontweight='bold', pad=15)
plt.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df["Free_Total_Ratio_Percent"], bins=30, color='#6C7A89', 
         edgecolor='white', linewidth=1.2, alpha=0.8)
plt.xlabel("Free/Total PSA Ratio (%)", fontweight='medium')
plt.ylabel("Count", fontweight='medium')
plt.title("Free/Total PSA Ratio Distribution", fontweight='bold', pad=15)
plt.axvline(10, color=REF_LINE_COLORS[10], linestyle='--', linewidth=2, 
            alpha=0.7, label='10% boundary')
plt.axvline(15, color=REF_LINE_COLORS[15], linestyle='--', linewidth=2, 
            alpha=0.7, label='15% boundary')
plt.axvline(25, color=REF_LINE_COLORS[25], linestyle='--', linewidth=2, 
            alpha=0.7, label='25% boundary')
plt.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.tight_layout()
plt.show()





# =========================
# 4. Log–log law validation
# =========================
plt.figure(figsize=(10, 7))
plt.scatter(
    np.log10(df["Free_PSA_nM"]),
    np.log10(df["Free_PSA_freq_Hz"]),
    s=30,
    alpha=0.65,
    label='Free PSA',
    color=HIST_COLORS[0],
    edgecolors='white',
    linewidth=0.5
)
plt.scatter(
    np.log10(df["Total_PSA_nM"]),
    np.log10(df["Total_PSA_freq_Hz"]),
    s=30,
    alpha=0.65,
    label='Total PSA',
    color=HIST_COLORS[1],
    edgecolors='white',
    linewidth=0.5
)
plt.xlabel("log(PSA concentration)", fontweight='medium', fontsize=14)
plt.ylabel("log(PSA frequency)", fontweight='medium', fontsize=14)
plt.title("Log–Log Relationship: Free and Total PSA", fontweight='bold', pad=15, fontsize=16)
plt.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()





# =========================
# 5. Risk Level Distribution Analysis (4 subplots)
# =========================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Risk Level Distribution Analysis', fontsize=16, fontweight='bold', y=0.995)

# Get risk level counts in order
risk_levels = ["High Risk", "Moderately High Risk", "Intermediate", "Low Risk"]
risk_counts = df["Risk_Level"].value_counts().reindex(risk_levels)


risk_levels_label = ["High Risk", "Moderately High Risk", "Intermediate Risk", "Low Risk"]

risk_levels1 = ["High", "Moderately High", "Intermediate", "Low "]

# Professional color list matching risk levels
risk_color_list = [RISK_COLORS[level] for level in risk_levels]

# Subplot 1: Bar chart of counts per risk level
ax1 = axes[0, 0]
bars = ax1.bar(risk_levels1, risk_counts.values, color=risk_color_list, 
               alpha=0.9, edgecolor='#2C2C2C', linewidth=1.2)
ax1.set_xlabel('Risk Level', fontweight='bold',fontsize=18)
ax1.set_ylabel('Number of Samples', fontweight='bold',fontsize=18)
ax1.set_title('Sample Count by Risk Level', fontweight='bold')
#ax1.tick_params(axis='x', rotation=45)
ax1.tick_params(axis='both', which='major',  labelsize=14 , width=1.5, length=6)
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')




# Subplot 2: Box plot of ratio distribution per risk level
ax2 = axes[0, 1]
ratio_data = [df[df["Risk_Level"] == level]["Free_Total_Ratio_Percent"].values for level in risk_levels]
bp = ax2.boxplot(ratio_data, labels=risk_levels1, patch_artist=True, 
                 widths=0.6, showmeans=True, meanline=True)
for patch, color in zip(bp['boxes'], risk_color_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
    patch.set_edgecolor('#2C2C2C')
    patch.set_linewidth(1.2)
# Style the median and mean lines
for element in ['whiskers', 'caps']:
    plt.setp(bp[element], color='#2C2C2C', linewidth=1.2)
plt.setp(bp['medians'], color='#FFFFFF', linewidth=2)
plt.setp(bp['means'], color='#F5F5F5', linewidth=1.5, linestyle='--')
plt.setp(bp['fliers'], color='#2C2C2C', marker='o', markersize=4, alpha=0.6)
ax2.axhline(10, color=REF_LINE_COLORS[10], linestyle='--', linewidth=2, 
            alpha=0.6, label='10% boundary')
ax2.axhline(15, color=REF_LINE_COLORS[15], linestyle='--', linewidth=2, 
            alpha=0.6, label='15% boundary')
ax2.axhline(25, color=REF_LINE_COLORS[25], linestyle='--', linewidth=2, 
            alpha=0.6, label='25% boundary')
ax2.set_xlabel('Risk Level', fontweight='bold', fontsize=18, labelpad=15)
ax2.set_ylabel('Free/Total PSA Ratio (%)', fontweight='bold',fontsize=18, labelpad=15)
ax2.set_title('Ratio Distribution by Risk Level', fontweight='bold', pad=10)
#ax2.tick_params(axis='x', rotation=0)
ax2.tick_params(axis='both', which='major',  labelsize=14 , width=1.5, length=6)

ax2.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9, fontsize=8)
ax2.grid(axis='y', alpha=0.3, linestyle='--')







# Subplot 3: Box plot of Free PSA concentration per risk level
ax3 = axes[1, 0]
free_psa_data = [df[df["Risk_Level"] == level]["Free_PSA_nM"].values for level in risk_levels]
bp3 = ax3.boxplot(free_psa_data, labels=risk_levels, patch_artist=True, 
                  widths=0.6, showmeans=True, meanline=True)
for patch, color in zip(bp3['boxes'], risk_color_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
    patch.set_edgecolor('#2C2C2C')
    patch.set_linewidth(1.2)
for element in ['whiskers', 'caps']:
    plt.setp(bp3[element], color='#2C2C2C', linewidth=1.2)
plt.setp(bp3['medians'], color='#FFFFFF', linewidth=2)
plt.setp(bp3['means'], color='#F5F5F5', linewidth=1.5, linestyle='--')
plt.setp(bp3['fliers'], color='#2C2C2C', marker='o', markersize=4, alpha=0.6)
ax3.set_xlabel('Risk Level', fontweight='medium')
ax3.set_ylabel('Free PSA Concentration (nM)', fontweight='medium')
ax3.set_title('Free PSA Concentration by Risk Level', fontweight='bold', pad=10)
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Subplot 4: Box plot of Total PSA concentration per risk level
ax4 = axes[1, 1]
total_psa_data = [df[df["Risk_Level"] == level]["Total_PSA_nM"].values for level in risk_levels]
bp4 = ax4.boxplot(total_psa_data, labels=risk_levels, patch_artist=True, 
                  widths=0.6, showmeans=True, meanline=True)
for patch, color in zip(bp4['boxes'], risk_color_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
    patch.set_edgecolor('#2C2C2C')
    patch.set_linewidth(1.2)
for element in ['whiskers', 'caps']:
    plt.setp(bp4[element], color='#2C2C2C', linewidth=1.2)
plt.setp(bp4['medians'], color='#FFFFFF', linewidth=2)
plt.setp(bp4['means'], color='#F5F5F5', linewidth=1.5, linestyle='--')
plt.setp(bp4['fliers'], color='#2C2C2C', marker='o', markersize=4, alpha=0.6)
ax4.set_xlabel('Risk Level', fontweight='medium')
ax4.set_ylabel('Total PSA Concentration (nM)', fontweight='medium')
ax4.set_title('Total PSA Concentration by Risk Level', fontweight='bold', pad=10)
ax4.tick_params(axis='x', rotation=45)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()
