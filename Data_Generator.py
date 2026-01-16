'''
This script generates synthetic PSA (Prostate-Specific Antigen) biomarker data.



What it does:
1. Generates concentrations: Creates Free PSA and Total PSA concentration values (in nM)
   within a specified range, ensuring Free PSA < Total PSA
   
2. Converts to frequency: Transforms concentration values to frequency measurements (Hz)
   using the conversion formula: frequency = (10^6.63) * (concentration_M^0.6)
   
3. Categorizes risk levels: Assigns risk levels based on Free/Total PSA ratio:
   - High Risk: ratio < 10%
   - Moderately High Risk: ratio 10-15%
   - Intermediate: ratio 15-25%
   - Low Risk: ratio > 25%
   
4. Prepares data: Organizes all data into a structured DataFrame and exports to CSV
   for further analysis or machine learning applications
   
'''

#import libraries
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# PSA concentration range (same for both Free and Total PSA)
PSA_RANGE = (0.01, 0.8)

# Number of samples per risk level (for balanced distribution)
N_PER_CLASS = 125  # 4 classes × 125 = 500 samples (or adjust to 100-150 for 400-600 range)

def freq_from_conc(conc_nM):
    '''
    Converts PSA concentration from nM to M and then to frequency using the conversion formula: frequency = (10^6.63) * (concentration_M^0.6)
    '''
    conc_M = conc_nM * 1e-9  # convert nM → M
    return (10 ** 6.63) * (conc_M ** 0.6)


data = []
sample_id = 1
def add_samples_by_ratio(ratio_min, ratio_max, label, n):
    """
    Generate samples with Free PSA / Total PSA ratio in specified range.
    Ratio is in percentage (e.g., 10 means 10%).
    Ensures free_psa < total_psa always.
    """
    global sample_id
    count = 0
    max_attempts = n * 10  # Prevent infinite loop
    attempts = 0
    
    while count < n and attempts < max_attempts:
        attempts += 1
        
        # Generate total PSA first
        total_psa = np.random.uniform(*PSA_RANGE)
        
        # Calculate min and max free PSA based on ratio constraints
        free_psa_min = (ratio_min / 100.0) * total_psa
        free_psa_max = min((ratio_max / 100.0) * total_psa, total_psa * 0.99)  # Ensure free < total
        
        # Only proceed if valid range exists
        if free_psa_min < free_psa_max:
            free_psa = np.random.uniform(free_psa_min, free_psa_max)
            
            # Calculate actual ratio
            ratio = (free_psa / total_psa) * 100
            
            # Verify ratio is in desired range
            # For "Low Risk" (ratio > 25%), use > instead of >=
            # For other ranges, use standard half-open intervals
            if ratio_min == 25:  # Low Risk case: ratio > 25%
                if ratio > ratio_min and ratio <= ratio_max:
                    valid = True
                else:
                    valid = False
            else:  # Other cases: ratio_min <= ratio < ratio_max
                if ratio_min <= ratio < ratio_max:
                    valid = True
                else:
                    valid = False
            
            if valid:
                data.append([
                    sample_id,
                    free_psa,
                    total_psa,
                    ratio,
                    label,
                    freq_from_conc(free_psa),
                    freq_from_conc(total_psa)
                ])
                sample_id += 1
                count += 1





# Generate samples for each risk level based on Free/Total PSA ratio
# Risk levels:
# - High Risk: ratio < 10%
# - Moderately High Risk: ratio 10-15%
# - Intermediate: ratio 15-25%
# - Low Risk: ratio > 25%

# HIGH RISK: ratio < 10%
add_samples_by_ratio(0, 10, "High Risk", N_PER_CLASS)

# MODERATELY HIGH RISK: ratio 10-15%
add_samples_by_ratio(10, 15, "Moderately High Risk", N_PER_CLASS)

# INTERMEDIATE: ratio 15-25%
add_samples_by_ratio(15, 25, "Intermediate", N_PER_CLASS)

# LOW RISK: ratio > 25% (but still free_psa < total_psa, so max ~99%)
add_samples_by_ratio(25, 99, "Low Risk", N_PER_CLASS)

# Create DataFrame
df = pd.DataFrame(
    data,
    columns=[
        "Sample_ID",
        "Free_PSA_nM",
        "Total_PSA_nM",
        "Free_Total_Ratio_Percent",
        "Risk_Level",
        "Free_PSA_freq_Hz",
        "Total_PSA_freq_Hz"
    ]
)


# Print the first 5 rows of the DataFrame and the value counts of the Risk_Level column
print(df.head())
print(df["Risk_Level"].value_counts())



# Export the DataFrame to a CSV file
#** You can Customize the name of your file
df.to_csv("Free_Total_PSA_frequency.csv", index=False, float_format="%.14f")
