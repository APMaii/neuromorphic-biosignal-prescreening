'''
Example workflow for running the SNN model on the PSA biomarker dataset.

you can go for two options:
1- Run the full pipeline
2- Run manually Each steps


Also after running the SNN trainin , you can save the model and load it later for testing or further analysis.

'''

#============================================================
'''          1- Run the full pipeline                     '''
#============================================================

#import the SNN_BioMarker class
from SNN_Core import SNN_BioMarker

# Ask the user to enter the path to the CSV file
path = input('Enter the path to the CSV file (Free_Total_PSA_frequency.csv): ')

# Create an instance of the SNN_BioMarker class
SNN_BioMarker_full_pipeline = SNN_BioMarker(path,dpi=600)

# Run the full pipeline

SNN_BioMarker_full_pipeline.run_full_pipeline(dt = 1e-2 , T = 5,a = 0.5,target_length=25,downsampled_signals=False,
split_ratio=0.8,batch_size=8,num_epochs=210,num_layers=3,num_hidden1=40,num_hidden2=20,num_hidden3=8,
beta=0.9,correct_rate=0.8,incorrect_rate=0.2,lr=1e-3,betas=(0.9, 0.999),generation_method='5',save_model=True,model_name='SNN_3_LAYER')

SNN_BioMarker_full_pipeline.visualize_signal_flow2(sample_idx=108)

#============================================================
'''          2- Run manually Each steps                     '''
#============================================================

#import the SNN_BioMarker class
from SNN_Core import SNN_BioMarker

# Ask the user to enter the path to the CSV file
path = input('Enter the path to the CSV file (Free_Total_PSA_frequency.csv): ')

# Create an instance of the SNN_BioMarker class
SNN_BioMarker_manual = SNN_BioMarker(path,dpi=600)

# Generate synaptic signals
SNN_BioMarker_manual.generate_synaptic_signals(dt = 1e-2 , T = 5,a = 0.5)

# Labeling y
SNN_BioMarker_manual.labeling_y()

# Convert to torch tensor
SNN_BioMarker_manual.convert_to_torch_tensor(downsampled_signals=False)

# Create dataset
SNN_BioMarker_manual.create_dataset()

# Split dataset
SNN_BioMarker_manual.split_dataset(split_ratio=0.8)

# Create data loaders
SNN_BioMarker_manual.create_data_loaders(batch_size=8)

#Create network
SNN_BioMarker_manual.create_network(num_layers=3,num_hidden1=40,num_hidden2=20,num_hidden3=8,beta=0.9)

# Create loss function
SNN_BioMarker_manual.create_loss_function(correct_rate=0.8,incorrect_rate=0.2)

# Create optimizer
SNN_BioMarker_manual.create_optimizer(lr=1e-3,betas=(0.9, 0.999))

# Training loop
SNN_BioMarker_manual.training_loop(num_epochs=210)

# Final evaluation
SNN_BioMarker_manual.final_evaluation()

# Plot loss curves
SNN_BioMarker_manual.plot_loss_curves()

# Plot confusion matrix
SNN_BioMarker_manual.plot_confusion_matrix()

# Test single prediction
SNN_BioMarker_manual.test_single_prediction()

# Visualize signal flow
SNN_BioMarker_manual.visualize_signal_flow2(sample_idx=108)

# Save model
SNN_BioMarker_manual.save_model(model_name='SNN_3_LAYER')







#============================================================
'''          3- Load model and test                     '''
#============================================================

from SNN_Core import SNN_BioMarker

# Ask the user to enter the path to the CSV file
data_path = input('Enter the path to the CSV file (Free_Total_PSA_frequency.csv): ')

# Create an instance of the SNN_BioMarker class
SNN_BioMarker_load_model = SNN_BioMarker(data_path,dpi=600)


model_path = input('Enter the path to the model (.pth): ')
SNN_BioMarker_load_model.load_model(model_path)


'''
Then you can run the following steps:
  - plot_loss_curves()
  - plot_confusion_matrix()
  - test_single_prediction()
  - visualize_signal_flow2()
  
  '''
SNN_BioMarker_load_model.plot_loss_curves()
SNN_BioMarker_load_model.plot_confusion_matrix()
SNN_BioMarker_load_model.test_single_prediction()
SNN_BioMarker_load_model.visualize_signal_flow2(sample_idx=9)

















