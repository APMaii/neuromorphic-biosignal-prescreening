#============================================================
# IMPORTING LIBRARIES
#============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import snntorch as snn
from snntorch import surrogate, utils
import snntorch.functional as SF
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import snntorch.spikeplot as splt
from IPython.display import HTML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib as mpl
import random

# ============================================================
# REPRODUCIBILITY SETTINGS
# ============================================================
# Set random seeds for reproducibility
REPRODUCIBILITY_SEED = 42

def set_seed(seed=REPRODUCIBILITY_SEED):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # PyTorch deterministic operations (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed} for reproducibility")

# Set seed globally at module import
set_seed(REPRODUCIBILITY_SEED)

# ============================================================
# PROFESSIONAL COLOR PALETTE
# ============================================================
# Professional color scheme for all visualizations
PROF_COLORS = {
    'fpsa': '#2563EB',           # Clear professional blue (easily distinguishable)
    'tpsa': '#DC2626',           # Clear professional red (easily distinguishable)
    'original': '#4A6FA5',       # Muted slate blue
    'downsampled': '#B87333',    # Muted terracotta
    'train': '#5A7D5A',          # Muted sage green
    'test': '#8B4A6B',           # Professional burgundy
    'hidden1': '#5A7D5A',        # Muted sage green
    'hidden2': '#6B8E7E',        # Muted teal
    'hidden3': '#7B9FA8',        # Muted blue-green
    'class0': '#4A6FA5',         # Muted slate blue (Low Risk)
    'class1': '#5A7D5A',         # Muted sage green (Intermediate)
    'class2': '#B87333',         # Muted terracotta (Moderately High Risk)
    'class3': '#8B4A6B',         # Professional burgundy (High Risk)
}

# Class colors list for output neurons
CLASS_COLORS = [PROF_COLORS['class0'], PROF_COLORS['class1'], 
                PROF_COLORS['class2'], PROF_COLORS['class3']]

# Professional colormaps for confusion matrices
CMAP_TRAIN = 'Blues'
CMAP_TEST = 'Purples'

# Set professional matplotlib style
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

# Create dataset class
class SpikeDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]





class CancerNet_1layer(nn.Module):
    def __init__(self,num_inputs=2,num_hidden1=60,num_outputs=4,beta=0.9):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Layer 1
        self.fc1 = nn.Linear(num_inputs, num_hidden1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Output layer
        self.fc2 = nn.Linear(num_hidden1, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
    
    def forward(self, x):
        """
        Forward pass through SNN.
        
        Args:
            x: Input of shape [num_steps, batch_size, num_inputs]
        
        Returns:
            spk4_rec: Output spikes [num_steps, batch_size, num_outputs]
            mem4_rec: Output membrane potentials [num_steps, batch_size, num_outputs]
        """
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record outputs
        spk2_rec = []
        mem2_rec = []
        
        # Iterate through time
        for step in range(x.shape[0]):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            

        
        return torch.stack(spk2_rec), torch.stack(mem2_rec)





class CancerNet_2layer(nn.Module):
    def __init__(self,num_inputs=2,num_hidden1=60,num_hidden2=20,num_outputs=4,beta=0.9):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Layer 1
        self.fc1 = nn.Linear(num_inputs, num_hidden1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Layer 2
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Output layer
        self.fc3 = nn.Linear(num_hidden2, num_outputs)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
    
    def forward(self, x):
        """
        Forward pass through SNN.
        
        Args:
            x: Input of shape [num_steps, batch_size, num_inputs]
        
        Returns:
            spk4_rec: Output spikes [num_steps, batch_size, num_outputs]
            mem4_rec: Output membrane potentials [num_steps, batch_size, num_outputs]
        """
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Record outputs
        spk3_rec = []
        mem3_rec = []
        
        # Iterate through time
        for step in range(x.shape[0]):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)
            

        
        return torch.stack(spk3_rec), torch.stack(mem3_rec)





class CancerNet_3layer(nn.Module):
    def __init__(self,num_inputs=2,num_hidden1=60,num_hidden2=20,num_hidden3=10,num_outputs=4,beta=0.9):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Layer 1
        self.fc1 = nn.Linear(num_inputs, num_hidden1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Layer 2
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Layer 3
        self.fc3 = nn.Linear(num_hidden2, num_hidden3)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Output layer
        self.fc4 = nn.Linear(num_hidden3, num_outputs)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
    
    def forward(self, x):
        """
        Forward pass through SNN.
        
        Args:
            x: Input of shape [num_steps, batch_size, num_inputs]
        
        Returns:
            spk4_rec: Output spikes [num_steps, batch_size, num_outputs]
            mem4_rec: Output membrane potentials [num_steps, batch_size, num_outputs]
        """
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        
        # Record outputs
        spk4_rec = []
        mem4_rec = []
        
        # Iterate through time
        for step in range(x.shape[0]):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            
            spk4_rec.append(spk4)
            mem4_rec.append(mem4)
            

        
        return torch.stack(spk4_rec), torch.stack(mem4_rec)



class SNN_BioMarker:
    
    def __init__(self,path):
        self.path = path
        if 'csv' in path.lower():
            self.df = pd.read_csv(path)
            
        elif path.lower() in ['xlsx','xls']:
            self.df = pd.read_excel(path)
            
        else:
            raise ValueError(f"Unsupported file format: {path}")

        #they used in function generate_synaptic_signals
        self.X_fPSA = []
        self.X_tPSA = []
        self.y = []
        self.N = 0
        self.dt= 0
        self.T= 0
        self.a= 0
        self.target_length=0

        self.X_fPSA_downsampled = None
        self.X_tPSA_downsampled = None

        self.y_numeric = None
        self.data_combined = None
        self.y_torch = None
        self.full_dataset = None

        self.train_dataset = None
        self.test_dataset = None

        self.batch_size = 0
        self.train_loader = None
        self.test_loader = None

        self.net = None

        #-----Flags-----
        self.generate_synaptic_signals_complete = False
        self.labeling_y_complete = False
        self.average_temporal_pooling_downsample_signal_complete = False
        self.convert_to_torch_tensor_complete = False
        self.create_dataset_complete = False
        self.split_dataset_complete = False
        self.create_data_loaders_complete = False
        self.create_network_complete = False
        self.create_loss_function_complete = False
        self.create_optimizer_complete = False
        self.training_loop_complete = False
        self.final_evaluation_complete = False

        print("Dataset loaded:")
        print(f"Columns: {self.df.columns.tolist()}")
        print(f"Shape: {self.df.shape}")
        print("\nFirst few rows:")
        print(self.df.head())
        
        # Store seed for this instance
        self.seed = REPRODUCIBILITY_SEED

    def set_seed(self, seed=42):
        """
        Set a custom random seed for reproducibility.
        Call this before creating network, splitting dataset, or training.
        
        Args:
            seed: Random seed value (default: 42)
        """
        self.seed = seed
        set_seed(seed)
        print(f"Random seed updated to {seed} for this instance")

    def generate_synaptic_signals(self,dt = 1e-3 , T = 20,a = 0.9 ,generation_method='1' ):

        if generation_method == '1':
            self._generate_synaptic_signals1(dt = dt , T = T,a = a)
        elif generation_method == '2':
            self._generate_synaptic_signals2(dt = dt , T = T,a = a)
        elif generation_method == '3':
            self._generate_synaptic_signals3(dt = dt , T = T,a = a)
        elif generation_method == '4':
            self._generate_synaptic_signals4(dt = dt , T = T,a = a)
        elif generation_method == '5':
            self._generate_synaptic_signals5(dt = dt , T = T,a = a)

        self.generate_synaptic_signals_complete = True


    
    
    def get_synaptic_signals(self):
        if not self.generate_synaptic_signals_complete:
            raise ValueError("Synaptic signals not generated yet. Please call generate_synaptic_signals() first.")
        return self.X_fPSA, self.X_tPSA, self.y

    def _generate_synaptic_signals1(self,dt = 1e-3 , T = 20,a = 0.9 ):
        '''
        This function generates the synaptic signals from the biomarker frequencies.
        It uses the following parameters:
        - dt: sampling time
        - T: duration
        - a: synaptic decay
        It returns the synaptic signals and the labels.

        '''
        self.N = int(T / dt)  # 20,000 timesteps
        self.dt= dt
        self.T= T
        self.a= a

        print(f"\n{'='*60}")
        print("GENERATING SYNAPTIC SIGNALS")
        print(f"{'='*60}")
        print(f"Initial timesteps: {self.N}")
        print(f"Duration: {T}s")
        print(f"Sampling rate: {dt}s")

        X_fPSA=[]
        X_tPSA=[]
        y=[]
        # Generate full-length synaptic signals
        for i in range(len(self.df)):
            f1 = self.df.loc[i, "Free_PSA_freq_Hz"]
            f2 = self.df.loc[i, "Total_PSA_freq_Hz"]
            
            # Generate spike trains
            spk1 = np.zeros(self.N)
            spk2 = np.zeros(self.N)
            
            step1 = int((1/f1) / self.dt) if f1 > 0 else self.N
            step2 = int((1/f2) / self.dt) if f2 > 0 else self.N
            
            spk1[0:self.N:step1] = 1
            spk2[0:self.N:step2] = 1
            
            # Synaptic filtering
            x1 = np.zeros(self.N)
            x2 = np.zeros(self.N)
            
            for t in range(1, self.N):
                x1[t] = spk1[t] + self.a * x1[t-1]
                x2[t] = spk2[t] + self.a * x2[t-1]
            
            X_fPSA.append(x1)
            X_tPSA.append(x2)
            y.append(self.df.loc[i, "Risk_Level"])

        self.X_fPSA = np.array(X_fPSA)
        self.X_tPSA = np.array(X_tPSA)
        self.y = np.array(y)

        print("Original signal shapes:")
        print(f"  X_fPSA: {self.X_fPSA.shape}")
        print(f"  X_tPSA: {self.X_tPSA.shape}")

        

    def _generate_synaptic_signals2(self, dt=1e-3, T=20, a=0.9):
        '''
        Improved version of generate_synaptic_signals that works correctly for larger dt values.
        This function generates the synaptic signals from the biomarker frequencies using a 
        time-based approach rather than step-based slicing.
        
        It uses the following parameters:
        - dt: sampling time
        - T: duration
        - a: synaptic decay
        It returns the synaptic signals and the labels.
        '''
        self.N = int(T / dt)  # Number of timesteps
        self.dt = dt
        self.T = T
        self.a = a

        print(f"\n{'='*60}")
        print("GENERATING SYNAPTIC SIGNALS (Version 2)")
        print(f"{'='*60}")
        print(f"Initial timesteps: {self.N}")
        print(f"Duration: {T}s")
        print(f"Sampling rate: {dt}s")

        X_fPSA = []
        X_tPSA = []
        y = []
        
        # Generate full-length synaptic signals
        for i in range(len(self.df)):
            f1 = self.df.loc[i, "Free_PSA_freq_Hz"]
            f2 = self.df.loc[i, "Total_PSA_freq_Hz"]
            
            # Generate spike trains using time-based approach
            spk1 = np.zeros(self.N)
            spk2 = np.zeros(self.N)
            
            # Generate spikes based on frequency
            if f1 > 0:
                period1 = 1.0 / f1  # Period in seconds
                # Generate spike times
                spike_times1 = np.arange(0, T, period1)
                # Convert spike times to indices
                spike_indices1 = np.round(spike_times1 / self.dt).astype(int)
                # Ensure indices are within bounds
                spike_indices1 = spike_indices1[spike_indices1 < self.N]
                spk1[spike_indices1] = 1
            else:
                # If frequency is 0, no spikes
                pass
            
            if f2 > 0:
                period2 = 1.0 / f2  # Period in seconds
                # Generate spike times
                spike_times2 = np.arange(0, T, period2)
                # Convert spike times to indices
                spike_indices2 = np.round(spike_times2 / self.dt).astype(int)
                # Ensure indices are within bounds
                spike_indices2 = spike_indices2[spike_indices2 < self.N]
                spk2[spike_indices2] = 1
            else:
                # If frequency is 0, no spikes
                pass
            
            # Synaptic filtering
            x1 = np.zeros(self.N)
            x2 = np.zeros(self.N)
            
            for t in range(1, self.N):
                x1[t] = spk1[t] + self.a * x1[t-1]
                x2[t] = spk2[t] + self.a * x2[t-1]
            
            X_fPSA.append(x1)
            X_tPSA.append(x2)
            y.append(self.df.loc[i, "Risk_Level"])

        self.X_fPSA = np.array(X_fPSA)
        self.X_tPSA = np.array(X_tPSA)
        self.y = np.array(y)

        print("Original signal shapes:")
        print(f"  X_fPSA: {self.X_fPSA.shape}")
        print(f"  X_tPSA: {self.X_tPSA.shape}")
        
        # Print some statistics for first sample
        if len(self.df) > 0:
            f1_sample = self.df.loc[0, "Free_PSA_freq_Hz"]
            f2_sample = self.df.loc[0, "Total_PSA_freq_Hz"]
            print("\nSignal statistics (first sample):")
            print(f"  Free PSA frequency: {f1_sample:.4f} Hz (expected ~{f1_sample*T:.1f} spikes in {T}s)")
            print(f"  Total PSA frequency: {f2_sample:.4f} Hz (expected ~{f2_sample*T:.1f} spikes in {T}s)")


    def _generate_synaptic_signals3(self, dt=1e-3, T=20, a=0.9):
        '''
        Improved version of generate_synaptic_signals that works correctly for larger dt values.
        This function generates the synaptic signals from the biomarker frequencies using a 
        time-based approach rather than step-based slicing.
        
        It uses the following parameters:
        - dt: sampling time
        - T: duration
        - a: synaptic decay
        It returns the synaptic signals and the labels.
        '''
        self.N = int(T / dt)  # Number of timesteps
        self.dt = dt
        self.T = T
        self.a = a

        print(f"\n{'='*60}")
        print("GENERATING SYNAPTIC SIGNALS (Version 2)")
        print(f"{'='*60}")
        print(f"Initial timesteps: {self.N}")
        print(f"Duration: {T}s")
        print(f"Sampling rate: {dt}s")

        X_fPSA = []
        X_tPSA = []
        y = []
        
        # Generate full-length synaptic signals
        for i in range(len(self.df)):
            f1 = self.df.loc[i, "Free_PSA_freq_Hz"]
            f2 = self.df.loc[i, "Total_PSA_freq_Hz"]
            
            # Generate spike trains using index-based approach (avoids accumulation errors)
            spk1 = np.zeros(self.N)
            spk2 = np.zeros(self.N)
            
            # Generate spikes based on frequency
            if f1 > 0:
                period1 = 1.0 / f1  # Period in seconds
                period_steps1 = period1 / self.dt  # Period in timesteps
                # Generate spike indices directly to avoid accumulation errors
                spike_indices1 = np.round(np.arange(0, self.N, period_steps1)).astype(int)
                # Ensure indices are within bounds and remove duplicates
                spike_indices1 = np.unique(spike_indices1[spike_indices1 < self.N])
                spk1[spike_indices1] = 1
            else:
                # If frequency is 0, no spikes
                pass
            
            if f2 > 0:
                period2 = 1.0 / f2  # Period in seconds
                period_steps2 = period2 / self.dt  # Period in timesteps
                # Generate spike indices directly to avoid accumulation errors
                spike_indices2 = np.round(np.arange(0, self.N, period_steps2)).astype(int)
                # Ensure indices are within bounds and remove duplicates
                spike_indices2 = np.unique(spike_indices2[spike_indices2 < self.N])
                spk2[spike_indices2] = 1
            else:
                # If frequency is 0, no spikes
                pass
            
            # Synaptic filtering
            x1 = np.zeros(self.N)
            x2 = np.zeros(self.N)
            
            for t in range(1, self.N):
                x1[t] = spk1[t] + self.a * x1[t-1]
                x2[t] = spk2[t] + self.a * x2[t-1]
            
            X_fPSA.append(x1)
            X_tPSA.append(x2)
            y.append(self.df.loc[i, "Risk_Level"])

        self.X_fPSA = np.array(X_fPSA)
        self.X_tPSA = np.array(X_tPSA)
        self.y = np.array(y)

        print("Original signal shapes:")
        print(f"  X_fPSA: {self.X_fPSA.shape}")
        print(f"  X_tPSA: {self.X_tPSA.shape}")
        print(f"  y: {self.y.shape}")
        # Print some statistics for first sample
        if len(self.df) > 0:
            f1_sample = self.df.loc[0, "Free_PSA_freq_Hz"]
            f2_sample = self.df.loc[0, "Total_PSA_freq_Hz"]
            print("\nSignal statistics (first sample):")
            print(f"  Free PSA frequency: {f1_sample:.4f} Hz (expected ~{f1_sample*T:.1f} spikes in {T}s)")
            print(f"  Total PSA frequency: {f2_sample:.4f} Hz (expected ~{f2_sample*T:.1f} spikes in {T}s)")



    def _generate_synaptic_signals4(self, dt=1e-3, T=20, a=0.9):
        # Parameters
        self.dt = dt
        self.T = T
        self.a = a
        
        self.N = int(np.round(self.T / self.dt))  # Number of samples

        X_fPSA = []
        X_tPSA = []
        y = []

        # Time vector for plotting
        #t = np.arange(self.N) * self.dt

        # Assuming 'data' is a pandas DataFrame with columns 'PSA_freq_Hz' and 'PSMA_freq_Hz'
        # Example: data = pd.read_csv('your_data.csv')

        for i in range(len(self.df)):
            f1 = self.df['Free_PSA_freq_Hz'].iloc[i]
            f2 = self.df['Total_PSA_freq_Hz'].iloc[i]
            
            # Generate spike trains for both frequencies
            spk1 = np.zeros(self.N)
            spk2 = np.zeros(self.N)
            
            # Place spikes at regular intervals based on frequency
            if f1 > 0:  # Avoid division by zero
                interval1 = int(np.round((1 / f1) / self.dt))
                spk1[::interval1] = 1
            
            if f2 > 0:  # Avoid division by zero
                interval2 = int(np.round((1 / f2) / self.dt))
                spk2[::interval2] = 1
            
            # Apply synaptic filter: H(z) = 1 / (1 - a*z^(-1))
            # This is a simple exponential decay filter
            x1 = signal.lfilter([1], [1, -self.a], spk1)
            x2 = signal.lfilter([1], [1, -self.a], spk2)

            X_fPSA.append(x1)
            X_tPSA.append(x2)
            y.append(self.df.loc[i, "Risk_Level"])
            
            # Visualization
            #fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot PSA signal
            #axes[0].plot(t, x1, linewidth=0.8)
            #axes[0].set_title(f'Sample {i} — PSA freq = {f1:.2f} Hz')
            #axes[0].set_xlabel('Time (s)')
            #axes[0].set_ylabel('Synaptic Signal')
            #axes[0].grid(True, alpha=0.3)
            
            # Plot PSMA signal
            #axes[1].plot(t, x2, linewidth=0.8)
            #axes[1].set_title(f'Sample {i} — PSMA freq = {f2:.2f} Hz')
            #axes[1].set_xlabel('Time (s)')
            #axes[1].set_ylabel('Synaptic Signal')
            #axes[1].grid(True, alpha=0.3)
            
            #plt.tight_layout()
            #plt.show()
            
        self.X_fPSA = np.array(X_fPSA)
        self.X_tPSA = np.array(X_tPSA)
        self.y = np.array(y)

        print("Original signal shapes:")
        print(f"  X_fPSA: {self.X_fPSA.shape}")
        print(f"  X_tPSA: {self.X_tPSA.shape}")
        print(f"  y: {self.y.shape}")


    def _generate_synaptic_signals5(self, dt=1e-3, T=20, a=0.9):
        # Parameters
        self.dt = dt
        self.T = T
        self.a = a
        
        self.N = int(np.round(self.T / self.dt))  # Number of samples

        # Time vector for plotting
        #t = np.arange(self.N) * self.dt

        X_fPSA = []
        X_tPSA = []
        y = []

        # Assuming 'data' is a pandas DataFrame
        for i in range(len(self.df)):
            f1 = self.df['Free_PSA_freq_Hz'].iloc[i]
            f2 = self.df['Total_PSA_freq_Hz'].iloc[i]
            
            # Generate spike trains
            spk1 = np.zeros(self.N)
            spk2 = np.zeros(self.N)
            
            if f1 > 0:
                interval1 = int(np.round((1 / f1) / self.dt))
                spk1[::interval1] = 1
            
            if f2 > 0:
                interval2 = int(np.round((1 / f2) / self.dt))
                spk2[::interval2] = 1
            
            # CORRECTED: Apply proper synaptic filter with exponential decay
            # Method 1: Use negative coefficient for decay
            # x(n) = a * x(n-1) + input(n)
            #x1 = signal.lfilter([1], [1, -self.a], spk1)
            #x2 = signal.lfilter([1], [1, -self.a], spk2)
            
            # Method 2 (RECOMMENDED): Manual implementation with clear decay
            # This ensures the signal decays back to zero
            x1_manual = np.zeros(self.N)
            x2_manual = np.zeros(self.N)
            
            for n in range(1, self.N):
                x1_manual[n] = self.a * x1_manual[n-1] + spk1[n]
                x2_manual[n] = self.a * x2_manual[n-1] + spk2[n]
            
            # Normalize to prevent accumulation (optional but recommended)
            # Scale by (1-a) to keep amplitude consistent
            x1_normalized = (1 - self.a) * x1_manual
            x2_normalized = (1 - self.a) * x2_manual

            '''
            
            # Visualization
            fig, axes = plt.subplots(3, 2, figsize=(14, 10))
            
            # Column 1: PSA
            axes[0, 0].stem(t, spk1, linefmt='b-', markerfmt='bo', basefmt='k-', use_line_collection=True)
            axes[0, 0].set_title(f'Sample {i} — PSA Spikes (freq = {f1:.2f} Hz)')
            axes[0, 0].set_ylabel('Spike')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xlim([0, min(5, self.T)])  # Show first 5 seconds
            
            axes[1, 0].plot(t, x1_manual, linewidth=1.0, color='blue')
            axes[1, 0].set_title('PSA Synaptic Current (Raw)')
            axes[1, 0].set_ylabel('Current')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xlim([0, min(5, self.T)])
            
            axes[2, 0].plot(t, x1_normalized, linewidth=1.0, color='blue')
            axes[2, 0].set_title('PSA Synaptic Current (Normalized)')
            axes[2, 0].set_xlabel('Time (s)')
            axes[2, 0].set_ylabel('Current')
            axes[2, 0].grid(True, alpha=0.3)
            axes[2, 0].set_xlim([0, min(5, self.T)])
            
            # Column 2: PSMA
            axes[0, 1].stem(t, spk2, linefmt='r-', markerfmt='ro', basefmt='k-', use_line_collection=True)
            axes[0, 1].set_title(f'Sample {i} — PSMA Spikes (freq = {f2:.2f} Hz)')
            axes[0, 1].set_ylabel('Spike')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xlim([0, min(5, self.T)])
            
            axes[1, 1].plot(t, x2_manual, linewidth=1.0, color='red')
            axes[1, 1].set_title('PSMA Synaptic Current (Raw)')
            axes[1, 1].set_ylabel('Current')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xlim([0, min(5, self.T)])
            
            axes[2, 1].plot(t, x2_normalized, linewidth=1.0, color='red')
            axes[2, 1].set_title('PSMA Synaptic Current (Normalized)')
            axes[2, 1].set_xlabel('Time (s)')
            axes[2, 1].set_ylabel('Current')
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].set_xlim([0, min(5, self.T)])
            
            plt.tight_layout()
            plt.show()
            '''

            X_fPSA.append(x1_normalized)
            X_tPSA.append(x2_normalized)
            y.append(self.df.loc[i, "Risk_Level"])

        self.X_fPSA = np.array(X_fPSA)
        self.X_tPSA = np.array(X_tPSA)
        self.y = np.array(y)

        print("Original signal shapes:")
        print(f"  X_fPSA: {self.X_fPSA.shape}")
        print(f"  X_tPSA: {self.X_tPSA.shape}")
        print(f"  y: {self.y.shape}")


    def labeling_y(self):
        if not self.generate_synaptic_signals_complete:
            raise ValueError("Synaptic signals not generated yet. Please call generate_synaptic_signals() first.")

        label_map = {'Low Risk': 0, 'Intermediate': 1, 'Moderately High Risk': 2, 'High Risk': 3}
        self.y_numeric = np.array([label_map[val] for val in self.y])

        # Check class distribution
        unique, counts = np.unique(self.y_numeric, return_counts=True)
        print(f"\n{'='*60}")
        print("CLASS DISTRIBUTION")
        print(f"{'='*60}")
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} samples ({count/len(self.y_numeric)*100:.1f}%)")
        print(f"Total samples: {len(self.y_numeric)}")

        self.labeling_y_complete = True


    def get_labels(self):
        if not self.labeling_y_complete:
            raise ValueError("Labels not labeled yet. Please call labeling_y() first.")

        return self.y_numeric

    def plot_synaptic_signals(self):
        '''
        This function plots the synaptic signals.
        It uses the following parameters:
        - dt: sampling time
        - T: duration
        - a: synaptic decay
        It returns the synaptic signals and the labels.
        '''

        if not self.generate_synaptic_signals_complete:
            raise ValueError("Synaptic signals not generated yet. Please call generate_synaptic_signals() first.")

        # Set seed for reproducible sample selection
        np.random.seed(REPRODUCIBILITY_SEED)
        num_samples_to_plot = 3
        target_sample = 108  # Always include sample 108
        
        # Validate target sample index
        if target_sample >= len(self.X_fPSA):
            raise ValueError(f"Target sample index {target_sample} is out of range. Dataset has {len(self.X_fPSA)} samples.")
        
        # Always include target sample, then randomly select remaining samples
        remaining_indices = [i for i in range(len(self.X_fPSA)) if i != target_sample]
        num_additional = num_samples_to_plot - 1
        additional_indices = np.random.choice(remaining_indices, num_additional, replace=False)
        sample_indices = np.sort(np.concatenate([[target_sample], additional_indices]))

        time = np.arange(self.N) * self.dt  # time axis in seconds

        fig, axes = plt.subplots(
            num_samples_to_plot, 1,
            figsize=(12, 6),
            sharex=True
        )

        fig.suptitle(
            "Example Synaptic Input Signals Generated from Biomarker Spiking Frequencies",
            fontsize=14
        )

        # Professional colors
        fpsa_color = PROF_COLORS['fpsa']
        tpsa_color = PROF_COLORS['tpsa']

        for i, idx in enumerate(sample_indices):
            axes[i].plot(time, self.X_fPSA[idx], color=fpsa_color, label="fPSA", linewidth=1.5, alpha=0.8)
            axes[i].plot(time, self.X_tPSA[idx], color=tpsa_color, label="tPSA", linewidth=1.5, alpha=0.8)
            axes[i].set_ylabel(f"Sample {idx}")
            if i == 0:
                axes[i].legend(loc="upper right")

        axes[-1].set_xlabel("Time (s)")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    
    def average_temporal_pooling_downsample_signal(self, target_length=25):
        """
        Downsample signal using average pooling to preserve information.
        
        Args:
            target_length: Desired number of timesteps
        
        Returns:
            Downsampled signal of shape (self.N, target_length)
        """
        if not self.generate_synaptic_signals_complete:
            raise ValueError("Synaptic signals not generated yet. Please call generate_synaptic_signals() first.")

        self.target_length= target_length
        num_samples, original_length = self.X_fPSA.shape
        window_size = self.N // self.target_length
        
        # Reshape and average over windows
        self.X_fPSA_downsampled = np.zeros((num_samples, self.target_length))
        
        for i in range(num_samples):
            for j in range(self.target_length):
                start_idx = j * window_size
                end_idx = start_idx + window_size
                if end_idx > original_length:
                    end_idx = original_length
                self.X_fPSA_downsampled[i, j] = np.mean(self.X_fPSA[i, start_idx:end_idx])
        
        


        num_samples, original_length = self.X_tPSA.shape

        # Reshape and average over windows
        self.X_tPSA_downsampled = np.zeros((num_samples, self.target_length))
        
        for i in range(num_samples):
            for j in range(self.target_length):
                start_idx = j * window_size
                end_idx = start_idx + window_size
                if end_idx > original_length:
                    end_idx = original_length
                self.X_tPSA_downsampled[i, j] = np.mean(self.X_tPSA[i, start_idx:end_idx])


        print(f"\n{'='*60}")
        print("DOWNSAMPLING SIGNALS")
        print(f"{'='*60}")
        print(f"Downsampled to {self.target_length} timesteps using average pooling")
        print(f"  X_fPSA_down: {self.X_fPSA_downsampled.shape}")
        print(f"  X_tPSA_down: {self.X_tPSA_downsampled.shape}")

        self.average_temporal_pooling_downsample_signal_complete = True





    def get_downsampled_signals(self):
        if not self.average_temporal_pooling_downsample_signal_complete:
            raise ValueError("Downsampled signals not generated yet. Please call average_temporal_pooling_downsample_signal() first.")

        return self.X_fPSA_downsampled, self.X_tPSA_downsampled



    def plot_comparison_downsampled_signals(self):
        if not self.average_temporal_pooling_downsample_signal_complete:
            raise ValueError("Downsampled signals not generated yet. Please call average_temporal_pooling_downsample_signal() first.")

        print(f"\n{'='*60}")
        print("COMPARING ORIGINAL vs DOWNSAMPLED SIGNALS")
        print(f"{'='*60}")

        # Select a few samples to visualize
        sample_indices = [0, 10, 50]  # Compare 3 different samples

        fig, axes = plt.subplots(len(sample_indices), 2, figsize=(16, 4*len(sample_indices)))

        # If only one sample, make axes 2D
        if len(sample_indices) == 1:
            axes = axes.reshape(1, -1)

        for idx, sample_idx in enumerate(sample_indices):
            # Get original and downsampled signals
            original_fpsa = self.X_fPSA[sample_idx]
            original_tpsa = self.X_tPSA[sample_idx]
            downsampled_fpsa = self.X_fPSA_downsampled[sample_idx]
            downsampled_tpsa = self.X_tPSA_downsampled[sample_idx]
            
            # Create time arrays
            time_original = np.arange(len(original_fpsa)) * self.dt  # in seconds
            time_downsampled = np.linspace(0, self.T, num=self.target_length)  # in seconds
            
            # Plot PSA signals
            ax1 = axes[idx, 0]
            ax1.plot(time_original, original_fpsa, color=PROF_COLORS['original'], 
                    alpha=0.6, linewidth=0.5, label='Original (20,000 steps)')
            ax1.plot(time_downsampled, downsampled_fpsa, color=PROF_COLORS['downsampled'], 
                    linestyle='-', marker='o', linewidth=2, markersize=6, label='Downsampled (25 steps)')
            ax1.set_xlabel('Time (seconds)', fontsize=11)
            ax1.set_ylabel('fPSA Signal Amplitude', fontsize=11)
            ax1.set_title(f'Sample {sample_idx}: fPSA Signal Comparison\n'
                        f'Original: {len(original_fpsa)} steps | Downsampled: {len(downsampled_fpsa)} steps', 
                        fontsize=12, fontweight='bold')
            ax1.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Plot tPSA signals
            ax2 = axes[idx, 1]
            ax2.plot(time_original, original_tpsa, color=PROF_COLORS['original'], 
                    alpha=0.6, linewidth=0.5, label='Original (20,000 steps)')
            ax2.plot(time_downsampled, downsampled_tpsa, color=PROF_COLORS['downsampled'], 
                    linestyle='-', marker='o', linewidth=2, markersize=6, label='Downsampled (25 steps)')
            ax2.set_xlabel('Time (seconds)', fontsize=11)
            ax2.set_ylabel('tPSA Signal Amplitude', fontsize=11)
            ax2.set_title(f'Sample {sample_idx}: tPSA Signal Comparison\n'
                        f'Original: {len(original_tpsa)} steps | Downsampled: {len(downsampled_tpsa)} steps', 
                        fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print statistical comparison
        print(f"\nStatistical Comparison (Sample {sample_indices[0]}):")
        print(f"{'='*60}")
        print("Original fPSA signal:")
        print(f"  Mean: {np.mean(original_fpsa):.6f}")
        print(f"  Std:  {np.std(original_fpsa):.6f}")
        print(f"  Min:  {np.min(original_fpsa):.6f}")
        print(f"  Max:  {np.max(original_fpsa):.6f}")
        print("\nDownsampled fPSA signal:")
        print(f"  Mean: {np.mean(downsampled_fpsa):.6f}")
        print(f"  Std:  {np.std(downsampled_fpsa):.6f}")
        print(f"  Min:  {np.min(downsampled_fpsa):.6f}")
        print(f"  Max:  {np.max(downsampled_fpsa):.6f}")
        print(f"\nMean difference: {abs(np.mean(original_fpsa) - np.mean(downsampled_fpsa)):.6f}")
        print(f"Compression ratio: {len(original_fpsa) / len(downsampled_fpsa):.1f}x")

    def convert_to_torch_tensor(self,downsampled_signals=True):

        if downsampled_signals:
            if not self.average_temporal_pooling_downsample_signal_complete:
                raise ValueError("Downsampled signals not generated yet. Please call average_temporal_pooling_downsample_signal() first.")
            if not self.labeling_y_complete:
                raise ValueError("Labels not labeled yet. Please call labeling_y() first.")
            
            X_fPSA_torch = torch.from_numpy(self.X_fPSA_downsampled).float()
            X_tPSA_torch = torch.from_numpy(self.X_tPSA_downsampled).float()
        else:
            if not self.generate_synaptic_signals_complete:
                raise ValueError("Synaptic signals not generated yet. Please call generate_synaptic_signals() first.")

            if not self.labeling_y_complete:
                raise ValueError("Labels not labeled yet. Please call labeling_y() first.")

            X_fPSA_torch = torch.from_numpy(self.X_fPSA).float()
            X_tPSA_torch = torch.from_numpy(self.X_tPSA).float()

        # Then we combine into (num_samples, num_steps, num_features)
        X_fPSA_3d = X_fPSA_torch.unsqueeze(-1)  # (num_samples, 25, 1)
        X_tPSA_3d = X_tPSA_torch.unsqueeze(-1)  # (num_samples, 25, 1)
        self.data_combined = torch.cat([X_fPSA_3d, X_tPSA_3d], dim=-1)  # (num_samples, 25, 2)

        #also we convert the labels to torch tensors
        self.y_torch = torch.from_numpy(self.y_numeric).long()

        print(f"\n{'='*60}")
        print("PYTORCH DATASET PREPARATION")
        print(f"{'='*60}")
        print(f"Combined data shape: {self.data_combined.shape}")
        print(f"Labels shape: {self.y_torch.shape}")

        self.convert_to_torch_tensor_complete = True


    def get_data_combined_torch(self):
        if not self.convert_to_torch_tensor_complete:
            raise ValueError("Data not converted to torch tensor yet. Please call convert_to_torch_tensor() first.")

        return self.data_combined
    
    def get_y_torch(self):
        if not self.convert_to_torch_tensor_complete:
            raise ValueError("Data not converted to torch tensor yet. Please call convert_to_torch_tensor() first.")

        return self.y_torch


    def create_dataset(self):
        if not self.convert_to_torch_tensor_complete:
            raise ValueError("Data not converted to torch tensor yet. Please call convert_to_torch_tensor() first.")

        self.full_dataset = SpikeDataset(self.data_combined, self.y_torch)

        print(f"\n{'='*60}")
        print("DATASET CREATION")
        print(f"{'='*60}")

        self.create_dataset_complete = True



    def get_full_dataset(self):
        if not self.create_dataset_complete:
            raise ValueError("Dataset not created yet. Please call create_dataset() first.")

        return self.full_dataset


    def split_dataset(self, split_ratio=0.8):
        if not self.create_dataset_complete:
            raise ValueError("Dataset not created yet. Please call create_dataset() first.")

        train_size = int(split_ratio * len(self.full_dataset))
        test_size = len(self.full_dataset) - train_size
        # Use seed for reproducible train/test split
        generator = torch.Generator()
        generator.manual_seed(REPRODUCIBILITY_SEED)
        self.train_dataset, self.test_dataset = random_split(
            self.full_dataset, [train_size, test_size],
            generator=generator
        )

        print("\nDataset split:")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Test: {len(self.test_dataset)} samples")
        self.split_dataset_complete = True



    def get_train_dataset(self):
        if not self.split_dataset_complete:
            raise ValueError("Dataset not split yet. Please call split_dataset() first.")

        return self.train_dataset

    def get_test_dataset(self):
        if not self.split_dataset_complete:
            raise ValueError("Dataset not split yet. Please call split_dataset() first.")

        return self.test_dataset

    def get_dataset(self):
        if not self.split_dataset_complete:
            raise ValueError("Dataset not split yet. Please call split_dataset() first.")

        return self.train_dataset, self.test_dataset


    def create_data_loaders(self, batch_size=32):
        if not self.split_dataset_complete:
            raise ValueError("Dataset not split yet. Please call split_dataset() first.")

        self.batch_size = batch_size
        
        # Create generator with seed for reproducible shuffling
        generator = torch.Generator()
        generator.manual_seed(REPRODUCIBILITY_SEED)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, 
                                       shuffle=True, generator=generator, 
                                       worker_init_fn=lambda worker_id: np.random.seed(REPRODUCIBILITY_SEED + worker_id))
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        print(f"\n{'='*60}")
        print("DATA LOADERS CREATION")
        print(f"{'='*60}")
        print(f"Train loader shape: {len(self.train_loader)}")
        print(f"Test loader shape: {len(self.test_loader)}")
        self.create_data_loaders_complete = True

    def get_train_loader(self):
        if not self.create_data_loaders_complete:
            raise ValueError("Data loaders not created yet. Please call create_data_loaders() first.")

        return self.train_loader

    def get_test_loader(self):
        if not self.create_data_loaders_complete:
            raise ValueError("Data loaders not created yet. Please call create_data_loaders() first.")

        return self.test_loader

    def get_data_loaders(self):
        if not self.create_data_loaders_complete:
            raise ValueError("Data loaders not created yet. Please call create_data_loaders() first.")

        return self.train_loader, self.test_loader


    def create_network(self,num_layers=1,num_hidden1=60,num_hidden2=20,num_hidden3=10,beta=0.9):
        if not self.create_data_loaders_complete:
            raise ValueError("Data loaders not created yet. Please call create_data_loaders() first.")

        # Set seed before network creation for reproducible weight initialization
        set_seed(REPRODUCIBILITY_SEED)

        if num_layers == 1:
            self.net = CancerNet_1layer(num_inputs=2,num_hidden1=num_hidden1,num_outputs=4,beta=beta)
        elif num_layers == 2:
            self.net = CancerNet_2layer(num_inputs=2,num_hidden1=num_hidden1,num_hidden2=num_hidden2,beta=beta)
        elif num_layers == 3:
            self.net = CancerNet_3layer(num_inputs=2,num_hidden1=num_hidden1,num_hidden2=num_hidden2,num_hidden3=num_hidden3,beta=beta)
        else:
            raise ValueError(f"Invalid number of layers: {num_layers}")

        # Initialize weights deterministically
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        self.net.apply(init_weights)

        print(f"\n{'='*60}")
        print("NETWORK ARCHITECTURE")
        print(f"{'='*60}")
        print(f"Number of layers: {num_layers}")
        if num_layers == 1: 
            print(f"Number of hidden neurons: {num_hidden1}")
        elif num_layers == 2:
            print(f"Number of hidden neurons: {num_hidden1}")
            print(f"Number of hidden neurons: {num_hidden2}")
        elif num_layers == 3:
            print(f"Number of hidden neurons: {num_hidden1}")
            print(f"Number of hidden neurons: {num_hidden2}")
            print(f"Number of hidden neurons: {num_hidden3}")
        else:
            raise ValueError(f"Invalid number of layers: {num_layers}")
        print(f"Beta: {beta}")
        print(f"Network weights initialized with seed: {REPRODUCIBILITY_SEED}")

        self.create_network_complete = True


    def get_network(self):
        if not self.create_network_complete:
            raise ValueError("Network not created yet. Please call create_network() first.")

        return self.net


    def create_loss_function(self,correct_rate=0.8,incorrect_rate=0.2):
        self.loss_fn = SF.mse_count_loss(correct_rate=correct_rate,incorrect_rate=incorrect_rate)
        self.create_loss_function_complete = True
        print(f"\n{'='*60}")
        print("LOSS FUNCTION Created")
        print(f"{'='*60}")


    def get_loss_function(self):
        if not self.create_loss_function_complete:
            raise ValueError("Loss function not created yet. Please call create_loss_function() first.")

        return self.loss_fn

    def create_optimizer(self,lr=1e-3,betas=(0.9, 0.999)):

        if not self.create_network_complete:
            raise ValueError("Network not created yet. Please call create_network() first.")

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=betas)

        print(f"\n{'='*60}")
        print("OPTIMIZER Created")
        print(f"{'='*60}")


        self.create_optimizer_complete = True

    def get_optimizer(self):
        if not self.create_optimizer_complete:
            raise ValueError("Optimizer not created yet. Please call create_optimizer() first.")

        return self.optimizer



    def _calculate_accuracy(self,data_loader):
        """Calculate accuracy on entire dataset."""
        
        if not self.create_data_loaders_complete:
            raise ValueError("Data loaders not created yet. Please call create_data_loaders() first.")

        if not self.create_network_complete:
            raise ValueError("Network not created yet. Please call create_network() first.")


        correct = 0
        total = 0
        
        with torch.no_grad():
            self.net.eval()
            for data, targets in data_loader:
                # Transpose: [batch, steps, features] -> [steps, batch, features]
                data_transposed = data.transpose(0, 1)
                
                # Reset hidden states
                utils.reset(self.net)
                
                # Forward pass
                spk_rec, _ = self.net(data_transposed)
                
                # Calculate accuracy
                acc = SF.accuracy_rate(spk_rec, targets)
                correct += (acc * targets.size(0)).item()
                total += targets.size(0)
        
        return 100 * correct / total

    def training_loop(self,num_epochs=400):

        if not self.create_data_loaders_complete:
            raise ValueError("Data loaders not created yet. Please call create_data_loaders() first.")

        if not self.create_network_complete:
            raise ValueError("Network not created yet. Please call create_network() first.")
            
        if not self.create_optimizer_complete:
            raise ValueError("Optimizer not created yet. Please call create_optimizer() first.")

        
        if not self.create_loss_function_complete:
            raise ValueError("Loss function not created yet. Please call create_loss_function() first.")

        
        self.num_epochs = num_epochs

        self.loss_hist = []
        self.test_loss_hist = []
        self.train_acc_hist = []
        self.test_acc_hist = []



        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}")

        
        for epoch in range(self.num_epochs):
            self.net.train()
            epoch_loss = 0
            
            for data, targets in self.train_loader:
                # Transpose: [batch, steps, features] -> [steps, batch, features]
                data_transposed = data.transpose(0, 1)
                
                # CRITICAL: Reset hidden states before forward pass
                utils.reset(self.net)
                
                # Forward pass
                spk_rec, mem_rec = self.net(data_transposed)
                
                # Calculate loss
                loss_val = self.loss_fn(spk_rec, targets)

                # Backward pass
                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()
                
                epoch_loss += loss_val.item()
            
            # Average training loss
            avg_train_loss = epoch_loss / len(self.train_loader)
            self.loss_hist.append(avg_train_loss)

            # Calculate test loss
            self.net.eval()
            test_loss = 0
            with torch.no_grad():
                for data, targets in self.test_loader:
                    data_transposed = data.transpose(0, 1)
                    utils.reset(self.net)
                    spk_rec, _ = self.net(data_transposed)
                    test_loss += self.loss_fn(spk_rec, targets).item()
            
            avg_test_loss = test_loss / len(self.test_loader)
            self.test_loss_hist.append(avg_test_loss)
            
            # Calculate accuracies every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                train_acc = self._calculate_accuracy(self.train_loader)
                test_acc = self._calculate_accuracy(self.test_loader)
                self.train_acc_hist.append(train_acc)
                self.test_acc_hist.append(test_acc)
                
                print(f"Epoch {epoch+1}/{self.num_epochs}")
                print(f"  Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
                print(f"  Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        

        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Final Train Accuracy: {self.train_acc_hist[-1]:.2f}%")
        print(f"Final Test Accuracy: {self.test_acc_hist[-1]:.2f}%")

        self.training_loop_complete = True


    def get_loss_hist(self):
        if not self.training_loop_complete:
            raise ValueError("Training loop not completed yet. Please call training_loop() first.")

        return self.loss_hist
    
    def get_test_loss_hist(self):
        if not self.training_loop_complete:
            raise ValueError("Training loop not completed yet. Please call training_loop() first.")

        return self.test_loss_hist
    
    def get_train_acc_hist(self):
        if not self.training_loop_complete:
            raise ValueError("Training loop not completed yet. Please call training_loop() first.")

        return self.train_acc_hist
    
    def get_test_acc_hist(self):
        if not self.training_loop_complete:
            raise ValueError("Training loop not completed yet. Please call training_loop() first.")

        return self.test_acc_hist

    def get_training_loop(self):
        if not self.training_loop_complete:
            raise ValueError("Training loop not completed yet. Please call training_loop() first.")

        return self.loss_hist, self.test_loss_hist, self.train_acc_hist, self.test_acc_hist


    def final_evaluation(self):
        if not self.training_loop_complete:
            raise ValueError("Training loop not completed yet. Please call training_loop() first.")

        print(f"\n{'='*60}")
        print("FINAL EVALUATION")
        print(f"{'='*60}")

        self.final_train_acc = self._calculate_accuracy(self.train_loader)
        self.final_test_acc = self._calculate_accuracy(self.test_loader)

        print(f"Final Train Accuracy: {self.final_train_acc:.2f}%")
        print(f"Final Test Accuracy: {self.final_test_acc:.2f}%")

        self.final_evaluation_complete = True

    def get_final_train_acc(self):
        if not self.final_evaluation_complete:
            raise ValueError("Final evaluation not completed yet. Please call final_evaluation() first.")

        return self.final_train_acc
    
    def get_final_test_acc(self):
        if not self.final_evaluation_complete:
            raise ValueError("Final evaluation not completed yet. Please call final_evaluation() first.")

        return self.final_test_acc

    def get_final_evaluation(self):
        if not self.final_evaluation_complete:
            raise ValueError("Final evaluation not completed yet. Please call final_evaluation() first.")

        return self.final_train_acc, self.final_test_acc



    def plot_loss_curves(self):
        if not self.training_loop_complete:
            raise ValueError("Training loop not completed yet. Please call training_loop() first.")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot loss curves
        axes[0].plot(self.loss_hist, label='Train Loss', linewidth=2, 
                    color=PROF_COLORS['train'], alpha=0.8)
        axes[0].plot(self.test_loss_hist, label='Test Loss', linewidth=2, 
                    color=PROF_COLORS['test'], alpha=0.8)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='medium')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='medium')
        axes[0].set_title('Training and Test Loss', fontsize=14, fontweight='bold')
        axes[0].legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        axes[0].grid(True, alpha=0.3, linestyle='--')

        # Plot accuracy curves
        epochs_recorded = [(i+1)*5 for i in range(len(self.train_acc_hist))]
        if 1 not in epochs_recorded:
            epochs_recorded = [1] + epochs_recorded
            
        axes[1].plot(epochs_recorded[:len(self.train_acc_hist)], self.train_acc_hist, 
                    marker='o', label='Train Accuracy', linewidth=2, 
                    color=PROF_COLORS['train'], alpha=0.8, markersize=5)
        axes[1].plot(epochs_recorded[:len(self.test_acc_hist)], self.test_acc_hist, 
                    marker='s', label='Test Accuracy', linewidth=2, 
                    color=PROF_COLORS['test'], alpha=0.8, markersize=5)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE!")
        print(f"{'='*60}")



    def _get_all_predictions(self,data_loader):
        """Get all predictions and true labels from data loader."""
        self.all_preds = []
        self.all_labels = []
        
        self.net.eval()
        with torch.no_grad():
            for data, targets in data_loader:
                # Transpose: [batch, steps, features] -> [steps, batch, features]
                data_transposed = data.transpose(0, 1)
                
                # Reset hidden states
                utils.reset(self.net)
                
                # Forward pass
                spk_rec, _ = self.net(data_transposed)
                
                # Get predictions (sum spikes over time, then argmax)
                spike_counts = spk_rec.sum(dim=0)  # [batch_size, num_classes]
                preds = spike_counts.argmax(dim=1)  # [batch_size]
                
                self.all_preds.extend(preds.cpu().numpy())
                self.all_labels.extend(targets.cpu().numpy())
        
        return np.array(self.all_preds), np.array(self.all_labels)

    def plot_confusion_matrix(self):
        if not self.final_evaluation_complete:
            raise ValueError("Final evaluation not completed yet. Please call final_evaluation() first.")

        # Get predictions for both train and test sets
        train_preds, train_labels = self._get_all_predictions(self.train_loader)
        test_preds, test_labels = self._get_all_predictions(self.test_loader)

        # Create confusion matrices
        cm_train = confusion_matrix(train_labels, train_preds)
        cm_test = confusion_matrix(test_labels, test_preds)

        # Class names
       # class_names = ['Low', 'Intermediate', 'Intermediate-High', 'High', 'Very High']
        class_names = ['Low Risk', 'Intermediate Risk', 'Moderately High Risk', 'High Risk']

        # Visualize confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Train confusion matrix
        sns.heatmap(cm_train, annot=True, fmt='d', cmap=CMAP_TRAIN, 
                    xticklabels=class_names, yticklabels=class_names,annot_kws={"size": 18, "weight": "bold"},
                    ax=axes[0], cbar_kws={'label': 'Count'}, 
                    linewidths=0.5, linecolor='white')
        axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Train Set Confusion Matrix\nAccuracy: {self.final_train_acc:.2f}%', 
                        fontsize=13, fontweight='bold')
        axes[0].tick_params(axis='both', labelsize=10)

        # Test confusion matrix
        sns.heatmap(cm_test, annot=True, fmt='d', cmap=CMAP_TEST, 
                    xticklabels=class_names, yticklabels=class_names,annot_kws={"size": 18, "weight": "bold"},
                    ax=axes[1], cbar_kws={'label': 'Count'}, 
                    linewidths=0.5, linecolor='white')
        axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Test Set Confusion Matrix\nAccuracy: {self.final_test_acc:.2f}%', 
                        fontsize=13, fontweight='bold')
        axes[1].tick_params(axis='both', labelsize=10)

        plt.tight_layout()
        plt.show()

        # Print detailed classification report
        print(f"\n{'='*60}")
        print("CLASSIFICATION REPORT - TEST SET")
        print(f"{'='*60}")
        print(classification_report(test_labels, test_preds, 
                                target_names=class_names, digits=3))

        # Print per-class accuracy
        print(f"\n{'='*60}")
        print("PER-CLASS ACCURACY - TEST SET")
        print(f"{'='*60}")
        for i, class_name in enumerate(class_names):
            # True positives for this class
            tp = cm_test[i, i]
            # Total samples of this class
            total = cm_test[i, :].sum()
            if total > 0:
                class_acc = (tp / total) * 100
                print(f"  {class_name:20s}: {class_acc:5.2f}% ({tp}/{total})")
            else:
                print(f"  {class_name:20s}: N/A (no samples)")

        print(f"\n{'='*60}")

    
    def test_single_prediction(self):
        if not self.final_evaluation_complete:
            raise ValueError("Final evaluation not completed yet. Please call final_evaluation() first.")

        print(f"\n{'='*60}")
        print("SINGLE SAMPLE PREDICTION TEST")
        print(f"{'='*60}")

        # Pick a random sample from test set (with seed for reproducibility)
        np.random.seed(REPRODUCIBILITY_SEED)
        random_idx = np.random.randint(0, len(self.test_dataset))
        sample_data, sample_label = self.test_dataset[random_idx]

        # Get original data info (before downsampling)
        # Find which original sample this corresponds to
        original_indices = self.test_dataset.indices
        original_sample_idx = original_indices[random_idx]

        # Get original biomarker values
        original_fpsa = self.df.loc[original_sample_idx, 'Free_PSA_nM']
        original_tpsa = self.df.loc[original_sample_idx, 'Total_PSA_nM']
        original_fpsa_freq = self.df.loc[original_sample_idx, 'Free_PSA_freq_Hz']
        original_tpsa_freq = self.df.loc[original_sample_idx, 'Total_PSA_freq_Hz']
        true_risk = self.df.loc[original_sample_idx, 'Risk_Level']

        print(f"\nRandomly selected test sample #{random_idx}")
        print(f"Original Sample ID: {original_sample_idx + 1}")
        print("\nOriginal Biomarker Values:")
        print(f"  fPSA concentration: {original_fpsa:.4f} nM")
        print(f"  tPSA concentration: {original_tpsa:.4f} nM")
        print(f"  fPSA frequency: {original_fpsa_freq:.4f} Hz")
        print(f"  tPSA frequency: {original_tpsa_freq:.4f} Hz")
        print(f"\nTrue Risk Level: {true_risk} (Class {sample_label.item()})")
        
        
        answer = input('are you ready? (y):')
        
        if answer.lower().strip() =='y':
            # Prepare data for prediction
            # Add batch dimension: [num_steps, num_features] -> [num_steps, 1, num_features]
            sample_data_batched = sample_data.unsqueeze(1)

            # Reset network and make prediction
            self.net.eval()
            with torch.no_grad():
                utils.reset(self.net)
                spk_out, mem_out = self.net(sample_data_batched)

            # Count spikes for each output neuron
            spike_counts = spk_out.sum(dim=0).squeeze()  # Sum over time dimension

            # Get predicted class
            predicted_class = spike_counts.argmax().item()

            # Get class names
            class_names = {0: 'Low Risk', 1: 'Intermediate', 2: 'Moderately High Risk', 3: 'High Risk'}

            print(f"\n{'='*60}")
            print("PREDICTION RESULTS")
            print(f"{'='*60}")
            print("\nSpike counts for each class:")
            for i, count in enumerate(spike_counts):
                marker = " ← PREDICTED" if i == predicted_class else ""
                true_marker = " ← TRUE LABEL" if i == sample_label.item() else ""
                print(f"  Class {i} ({class_names[i]:18s}): {count.item():6.2f} spikes{marker}{true_marker}")

            print(f"\nPredicted Risk Level: {class_names[predicted_class]} (Class {predicted_class})")
            print(f"True Risk Level:      {true_risk} (Class {sample_label.item()})")
            
            
            answer2= input('are you want to analyze results? (y):')
            
            if answer2.lower().strip()=='y':
                if predicted_class == sample_label.item():
                    print("\n✅ CORRECT PREDICTION!")
                else:
                    print("\n❌ INCORRECT PREDICTION")
                    print(f"   Model predicted: {class_names[predicted_class]}")
                    print(f"   Actual label: {true_risk}")

                # Calculate confidence (percentage of total spikes)
                total_spikes = spike_counts.sum().item()
                if total_spikes > 0:
                    confidence = (spike_counts[predicted_class].item() / total_spikes) * 100
                    print(f"\nPrediction confidence: {confidence:.1f}% of total spikes")
                else:
                    print("\nWarning: No spikes produced (network might need more training)")

                print(f"\n{'='*60}")
                
                
            else:
                print('Ok!')


        else:
            print('Ok!')

    def visualize_signal_flow(self, sample_idx=None, show_intermediate=True):
        """
        Visualize how signals flow through the network:
        - Input signals (fPSA and tPSA) entering the network
        - Intermediate layer outputs (if any)
        - Final output signals from output neurons
        
        Args:
            sample_idx: Index of sample to visualize (None = random from test set)
            show_intermediate: Whether to show intermediate layer outputs
        """
        if not self.final_evaluation_complete:
            raise ValueError("Network must be trained first. Please complete training and final_evaluation() first.")
        
        # Select sample
        if sample_idx is None:
            # Set seed for reproducible random selection
            np.random.seed(REPRODUCIBILITY_SEED)
            random_idx = np.random.randint(0, len(self.test_dataset))
            sample_data, sample_label = self.test_dataset[random_idx]
            original_indices = self.test_dataset.indices
            original_sample_idx = original_indices[random_idx]
        else:
            if sample_idx < len(self.test_dataset):
                sample_data, sample_label = self.test_dataset[sample_idx]
                original_indices = self.test_dataset.indices
                original_sample_idx = original_indices[sample_idx]
            else:
                # Use original dataset directly
                original_sample_idx = sample_idx
                if self.average_temporal_pooling_downsample_signal_complete:
                    sample_data = torch.from_numpy(np.array([
                        self.X_fPSA_downsampled[original_sample_idx],
                        self.X_tPSA_downsampled[original_sample_idx]
                    ])).float().T.unsqueeze(0)  # Shape: [1, timesteps, 2]
                    sample_data = sample_data.squeeze(0)  # [timesteps, 2]
                else:
                    sample_data = torch.from_numpy(np.array([
                        self.X_fPSA[original_sample_idx],
                        self.X_tPSA[original_sample_idx]
                    ])).float().T  # Shape: [timesteps, 2]
                sample_label = torch.tensor(self.y_numeric[original_sample_idx])
        
        # Get original biomarker info
        f1 = self.df.loc[original_sample_idx, "Free_PSA_freq_Hz"]
        f2 = self.df.loc[original_sample_idx, "Total_PSA_freq_Hz"]
        true_risk = self.df.loc[original_sample_idx, "Risk_Level"]
        
        # Get input signals (convert to numpy)
        if isinstance(sample_data, torch.Tensor):
            input_fpsa = sample_data[:, 0].cpu().numpy()
            input_tpsa = sample_data[:, 1].cpu().numpy()
            sample_data_tensor = sample_data
        else:
            input_fpsa = sample_data[:, 0]
            input_tpsa = sample_data[:, 1]
            sample_data_tensor = torch.from_numpy(sample_data).float()
        
        # Prepare data for network: [timesteps, features] -> [timesteps, 1, features]
        sample_data_batched = sample_data_tensor.unsqueeze(1)  # [timesteps, 1, features]
        
        # Determine network structure
        num_layers = 1
        if isinstance(self.net, CancerNet_3layer):
            num_layers = 3
        elif isinstance(self.net, CancerNet_2layer):
            num_layers = 2
        
        # Forward pass with intermediate outputs
        self.net.eval()
        with torch.no_grad():
            utils.reset(self.net)
            
            # Initialize states
            mem1 = self.net.lif1.init_leaky()
            if num_layers >= 2:
                mem2 = self.net.lif2.init_leaky()
            if num_layers >= 3:
                mem3 = self.net.lif3.init_leaky()
            mem_out = self.net.lif2.init_leaky() if num_layers == 1 else (self.net.lif3.init_leaky() if num_layers == 2 else self.net.lif4.init_leaky())
            
            # Storage for outputs
            hidden1_outputs = []
            hidden2_outputs = None
            hidden3_outputs = None
            if num_layers >= 2:
                hidden2_outputs = []
            if num_layers >= 3:
                hidden3_outputs = []
            output_spikes = []
            output_mem = []
            
            # Forward pass through time
            for step in range(sample_data_batched.shape[0]):
                x_step = sample_data_batched[step]  # [1, features]
                
                # Layer 1
                cur1 = self.net.fc1(x_step)
                spk1, mem1 = self.net.lif1(cur1, mem1)
                hidden1_outputs.append(spk1.squeeze().cpu().numpy())
                
                if num_layers == 1:
                    # Direct to output
                    cur_out = self.net.fc2(spk1)
                    spk_out, mem_out = self.net.lif2(cur_out, mem_out)
                elif num_layers == 2:
                    # Layer 2
                    cur2 = self.net.fc2(spk1)
                    spk2, mem2 = self.net.lif2(cur2, mem2)
                    hidden2_outputs.append(spk2.squeeze().cpu().numpy())
                    
                    # Output
                    cur_out = self.net.fc3(spk2)
                    spk_out, mem_out = self.net.lif3(cur_out, mem_out)
                else:  # num_layers == 3
                    # Layer 2
                    cur2 = self.net.fc2(spk1)
                    spk2, mem2 = self.net.lif2(cur2, mem2)
                    hidden2_outputs.append(spk2.squeeze().cpu().numpy())
                    
                    # Layer 3
                    cur3 = self.net.fc3(spk2)
                    spk3, mem3 = self.net.lif3(cur3, mem3)
                    hidden3_outputs.append(spk3.squeeze().cpu().numpy())
                    
                    # Output
                    cur_out = self.net.fc4(spk3)
                    spk_out, mem_out = self.net.lif4(cur_out, mem_out)
                
                output_spikes.append(spk_out.squeeze().cpu().numpy())
                output_mem.append(mem_out.squeeze().cpu().numpy())
            
            # Convert to numpy arrays
            hidden1_outputs = np.array(hidden1_outputs)
            if hidden2_outputs is not None:
                hidden2_outputs = np.array(hidden2_outputs)
            if hidden3_outputs is not None:
                hidden3_outputs = np.array(hidden3_outputs)
            output_spikes = np.array(output_spikes)
            output_mem = np.array(output_mem)
        
        # Create time axis based on actual data length
        actual_length = len(input_fpsa)
        if self.average_temporal_pooling_downsample_signal_complete:
            # Downsampled data: create time axis with target_length points
            time = np.linspace(0, self.T, num=actual_length)
        else:
            # Original data: create time axis with actual timesteps
            time = np.arange(actual_length) * self.dt
        
        # Create visualization
        num_plots = 3 + (1 if show_intermediate and hidden1_outputs is not None else 0)
        if num_layers >= 2 and show_intermediate:
            num_plots += 1
        if num_layers >= 3 and show_intermediate:
            num_plots += 1
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(14, 3*num_plots))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot 1: Input signals
        ax = axes[plot_idx]
        ax.plot(time, input_fpsa, color=PROF_COLORS['fpsa'], linewidth=1.5, 
               label=f'fPSA Input (freq: {f1:.4f} Hz)', alpha=0.8)
        ax.plot(time, input_tpsa, color=PROF_COLORS['tpsa'], linewidth=1.5, 
               label=f'tPSA Input (freq: {f2:.4f} Hz)', alpha=0.8)
        ax.set_ylabel('Input Signal\nAmplitude', fontsize=11, fontweight='bold')
        sample_label_val = sample_label.item() if isinstance(sample_label, torch.Tensor) else int(sample_label)
        ax.set_title(f'INPUT SIGNALS - Sample {original_sample_idx}\n'
                    f'True Label: {true_risk} (Class {sample_label_val})', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        plot_idx += 1
        
        # Plot intermediate layers if requested
        if show_intermediate and hidden1_outputs is not None:
            ax = axes[plot_idx]
            # Show average activity of hidden layer 1
            if hidden1_outputs.ndim == 2:
                avg_hidden1 = np.mean(hidden1_outputs, axis=1)
                ax.plot(time, avg_hidden1, color=PROF_COLORS['hidden1'], linewidth=1.5, 
                       label='Hidden Layer 1 (avg activity)', alpha=0.8)
                ax.fill_between(time, 0, avg_hidden1, alpha=0.3, color=PROF_COLORS['hidden1'])
            ax.set_ylabel('Hidden Layer 1\nActivity', fontsize=11, fontweight='bold')
            ax.set_title('INTERMEDIATE LAYER 1 OUTPUT', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            plot_idx += 1
            
            if num_layers >= 2 and hidden2_outputs is not None:
                ax = axes[plot_idx]
                if hidden2_outputs.ndim == 2:
                    avg_hidden2 = np.mean(hidden2_outputs, axis=1)
                    ax.plot(time, avg_hidden2, color=PROF_COLORS['hidden2'], linewidth=1.5, 
                           label='Hidden Layer 2 (avg activity)', alpha=0.8)
                    ax.fill_between(time, 0, avg_hidden2, alpha=0.3, color=PROF_COLORS['hidden2'])
                ax.set_ylabel('Hidden Layer 2\nActivity', fontsize=11, fontweight='bold')
                ax.set_title('INTERMEDIATE LAYER 2 OUTPUT', fontsize=12, fontweight='bold')
                ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, framealpha=0.9)
                ax.grid(True, alpha=0.3, linestyle='--')
                plot_idx += 1
            
            if num_layers >= 3 and hidden3_outputs is not None:
                ax = axes[plot_idx]
                if hidden3_outputs.ndim == 2:
                    avg_hidden3 = np.mean(hidden3_outputs, axis=1)
                    ax.plot(time, avg_hidden3, color=PROF_COLORS['hidden3'], linewidth=1.5, 
                           label='Hidden Layer 3 (avg activity)', alpha=0.8)
                    ax.fill_between(time, 0, avg_hidden3, alpha=0.3, color=PROF_COLORS['hidden3'])
                ax.set_ylabel('Hidden Layer 3\nActivity', fontsize=11, fontweight='bold')
                ax.set_title('INTERMEDIATE LAYER 3 OUTPUT', fontsize=12, fontweight='bold')
                ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, framealpha=0.9)
                ax.grid(True, alpha=0.3, linestyle='--')
                plot_idx += 1
        
        # Plot 2: Output spikes
        ax = axes[plot_idx]
        class_names = ['Low Risk', 'Intermediate','Moderately High Risk' ,'High Risk']
        
        for i in range(output_spikes.shape[1]):
            ax.plot(time, output_spikes[:, i], linewidth=1.5, 
                   label=f'Class {i} ({class_names[i]})', color=CLASS_COLORS[i], alpha=0.7)
        
        # Highlight predicted class
        spike_counts = np.sum(output_spikes, axis=0)
        predicted_class = np.argmax(spike_counts)
        ax.axvspan(time[0], time[-1], alpha=0.1, color=CLASS_COLORS[predicted_class], 
                  label=f'Predicted: {class_names[predicted_class]}')
        
        ax.set_ylabel('Output Spikes\n(Count per timestep)', fontsize=11, fontweight='bold')
        ax.set_title('OUTPUT LAYER - Spike Activity Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        plot_idx += 1
        
        # Plot 3: Output membrane potentials
        ax = axes[plot_idx]
        for i in range(output_mem.shape[1]):
            ax.plot(time, output_mem[:, i], linewidth=1.5, 
                   label=f'Class {i} ({class_names[i]})', color=CLASS_COLORS[i], alpha=0.7)
        
        ax.set_ylabel('Membrane Potential', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_title('OUTPUT LAYER - Membrane Potentials', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n{'='*60}")
        print("SIGNAL FLOW SUMMARY")
        print(f"{'='*60}")
        print(f"Sample: {original_sample_idx}")
        print(f"fPSA Frequency: {f1:.4f} Hz")
        print(f"tPSA Frequency: {f2:.4f} Hz")
        sample_label_val = sample_label.item() if isinstance(sample_label, torch.Tensor) else int(sample_label)
        print(f"True Label: {true_risk} (Class {sample_label_val})")
        print(f"\nOutput Spike Counts:")
        for i, count in enumerate(spike_counts):
            marker = " ← PREDICTED" if i == predicted_class else ""
            true_marker = " ← TRUE" if i == sample_label_val else ""
            print(f"  Class {i} ({class_names[i]:18s}): {count:6.0f} spikes{marker}{true_marker}")
        print(f"\nPredicted Class: {class_names[predicted_class]} (Class {predicted_class})")
        print(f"{'='*60}")

    def visualize_signal_flow2(self, sample_idx=None, max_neurons_per_fig=10):
        """
        Visualize membrane potential and spike output for EACH individual neuron:
        - Input neurons (fPSA and tPSA)
        - Each neuron in hidden layers (membrane potential + spikes)
        - Each neuron in output layer (membrane potential + spikes for each class)
        
        Args:
            sample_idx: Index of sample to visualize (None = random from test set)
            max_neurons_per_fig: Maximum number of neurons to show per figure
        """
        if not self.final_evaluation_complete:
            raise ValueError("Network must be trained first. Please complete training and final_evaluation() first.")
        
        # Select sample (same logic as visualize_signal_flow)
        if sample_idx is None:
            # Set seed for reproducible random selection
            np.random.seed(REPRODUCIBILITY_SEED)
            random_idx = np.random.randint(0, len(self.test_dataset))
            sample_data, sample_label = self.test_dataset[random_idx]
            original_indices = self.test_dataset.indices
            original_sample_idx = original_indices[random_idx]
        else:
            if sample_idx < len(self.test_dataset):
                sample_data, sample_label = self.test_dataset[sample_idx]
                original_indices = self.test_dataset.indices
                original_sample_idx = original_indices[sample_idx]
            else:
                original_sample_idx = sample_idx
                if self.average_temporal_pooling_downsample_signal_complete:
                    sample_data = torch.from_numpy(np.array([
                        self.X_fPSA_downsampled[original_sample_idx],
                        self.X_tPSA_downsampled[original_sample_idx]
                    ])).float().T.unsqueeze(0)
                    sample_data = sample_data.squeeze(0)
                else:
                    sample_data = torch.from_numpy(np.array([
                        self.X_fPSA[original_sample_idx],
                        self.X_tPSA[original_sample_idx]
                    ])).float().T
                sample_label = torch.tensor(self.y_numeric[original_sample_idx])
        
        # Get original biomarker info
        f1 = self.df.loc[original_sample_idx, "Free_PSA_freq_Hz"]
        f2 = self.df.loc[original_sample_idx, "Total_PSA_freq_Hz"]
        true_risk = self.df.loc[original_sample_idx, "Risk_Level"]
        
        # Get input signals
        if isinstance(sample_data, torch.Tensor):
            input_fpsa = sample_data[:, 0].cpu().numpy()
            input_tpsa = sample_data[:, 1].cpu().numpy()
            sample_data_tensor = sample_data
        else:
            input_fpsa = sample_data[:, 0]
            input_tpsa = sample_data[:, 1]
            sample_data_tensor = torch.from_numpy(sample_data).float()
        
        # Prepare data for network
        sample_data_batched = sample_data_tensor.unsqueeze(1)  # [timesteps, 1, features]
        
        # Determine network structure
        num_layers = 1
        if isinstance(self.net, CancerNet_3layer):
            num_layers = 3
        elif isinstance(self.net, CancerNet_2layer):
            num_layers = 2
        
        # Forward pass with detailed neuron tracking
        self.net.eval()
        with torch.no_grad():
            utils.reset(self.net)
            
            # Initialize states
            mem1 = self.net.lif1.init_leaky()
            if num_layers >= 2:
                mem2 = self.net.lif2.init_leaky()
            if num_layers >= 3:
                mem3 = self.net.lif3.init_leaky()
            mem_out = self.net.lif2.init_leaky() if num_layers == 1 else (self.net.lif3.init_leaky() if num_layers == 2 else self.net.lif4.init_leaky())
            
            # Storage for all neuron activities
            hidden1_spikes = []
            hidden1_mem = []
            hidden2_spikes = None
            hidden2_mem = None
            hidden3_spikes = None
            hidden3_mem = None
            if num_layers >= 2:
                hidden2_spikes = []
                hidden2_mem = []
            if num_layers >= 3:
                hidden3_spikes = []
                hidden3_mem = []
            output_spikes = []
            output_mem = []
            
            # Forward pass through time
            for step in range(sample_data_batched.shape[0]):
                x_step = sample_data_batched[step]  # [1, features]
                
                # Layer 1
                cur1 = self.net.fc1(x_step)
                spk1, mem1 = self.net.lif1(cur1, mem1)
                hidden1_spikes.append(spk1.squeeze().cpu().numpy())
                hidden1_mem.append(mem1.squeeze().cpu().numpy())
                
                if num_layers == 1:
                    cur_out = self.net.fc2(spk1)
                    spk_out, mem_out = self.net.lif2(cur_out, mem_out)
                elif num_layers == 2:
                    cur2 = self.net.fc2(spk1)
                    spk2, mem2 = self.net.lif2(cur2, mem2)
                    hidden2_spikes.append(spk2.squeeze().cpu().numpy())
                    hidden2_mem.append(mem2.squeeze().cpu().numpy())
                    
                    cur_out = self.net.fc3(spk2)
                    spk_out, mem_out = self.net.lif3(cur_out, mem_out)
                else:  # num_layers == 3
                    cur2 = self.net.fc2(spk1)
                    spk2, mem2 = self.net.lif2(cur2, mem2)
                    hidden2_spikes.append(spk2.squeeze().cpu().numpy())
                    hidden2_mem.append(mem2.squeeze().cpu().numpy())
                    
                    cur3 = self.net.fc3(spk2)
                    spk3, mem3 = self.net.lif3(cur3, mem3)
                    hidden3_spikes.append(spk3.squeeze().cpu().numpy())
                    hidden3_mem.append(mem3.squeeze().cpu().numpy())
                    
                    cur_out = self.net.fc4(spk3)
                    spk_out, mem_out = self.net.lif4(cur_out, mem_out)
                
                output_spikes.append(spk_out.squeeze().cpu().numpy())
                output_mem.append(mem_out.squeeze().cpu().numpy())
            
            # Convert to numpy arrays
            hidden1_spikes = np.array(hidden1_spikes)  # [timesteps, num_neurons]
            hidden1_mem = np.array(hidden1_mem)
            if hidden2_spikes is not None:
                hidden2_spikes = np.array(hidden2_spikes)
                hidden2_mem = np.array(hidden2_mem)
            if hidden3_spikes is not None:
                hidden3_spikes = np.array(hidden3_spikes)
                hidden3_mem = np.array(hidden3_mem)
            output_spikes = np.array(output_spikes)  # [timesteps, num_outputs]
            output_mem = np.array(output_mem)
        
        # Create time axis
        actual_length = len(input_fpsa)
        if self.average_temporal_pooling_downsample_signal_complete:
            time = np.linspace(0, self.T, num=actual_length)
        else:
            time = np.arange(actual_length) * self.dt
        
        sample_label_val = sample_label.item() if isinstance(sample_label, torch.Tensor) else int(sample_label)
        class_names = ['Low Risk', 'Intermediate','Moderately High Risk', 'High Risk']
        
        # ========== FIGURE 1: INPUT NEURONS ==========
        fig1, axes1 = plt.subplots(2, 1, figsize=(14, 8))
        fig1.suptitle(f'INPUT NEURONS - Sample {original_sample_idx}\n'
                      f'True Label: {true_risk} (Class {sample_label_val})', 
                      fontsize=14, fontweight='bold')
        
        # PSA Input
        ax = axes1[0]
        ax.plot(time, input_fpsa, color=PROF_COLORS['fpsa'], linewidth=2, 
               label=f'fPSA Signal (freq: {f1:.4f} Hz)', alpha=0.8)
        ax.set_ylabel('fPSA Signal\nAmplitude', fontsize=11, fontweight='bold')
        ax.set_title('fPSA Input Neuron', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # tPSA Input
        ax = axes1[1]
        ax.plot(time, input_tpsa, color=PROF_COLORS['tpsa'], linewidth=2, 
               label=f'tPSA Signal (freq: {f2:.4f} Hz)', alpha=0.8)
        ax.set_ylabel('tPSA Signal\nAmplitude', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_title('tPSA Input Neuron', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.show()
        
        # ========== FIGURE 2: HIDDEN LAYER 1 NEURONS ==========
        num_hidden1 = hidden1_spikes.shape[1]
        num_figs_h1 = int(np.ceil(num_hidden1 / max_neurons_per_fig))
        
        for fig_idx in range(num_figs_h1):
            start_idx = fig_idx * max_neurons_per_fig
            end_idx = min(start_idx + max_neurons_per_fig, num_hidden1)
            num_neurons_this_fig = end_idx - start_idx
            
            fig, axes = plt.subplots(num_neurons_this_fig, 2, figsize=(16, 3*num_neurons_this_fig))
            if num_neurons_this_fig == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'HIDDEN LAYER 1 - Neurons {start_idx} to {end_idx-1}\n'
                        f'Sample {original_sample_idx}', 
                        fontsize=14, fontweight='bold')
            
            for i, neuron_idx in enumerate(range(start_idx, end_idx)):
                # Membrane potential
                ax = axes[i, 0]
                ax.plot(time, hidden1_mem[:, neuron_idx], color=PROF_COLORS['hidden1'], 
                       linewidth=1.5, alpha=0.8)
                ax.set_ylabel(f'Neuron {neuron_idx}\nMembrane Pot.', fontsize=10, fontweight='bold')
                if i == num_neurons_this_fig - 1:
                    ax.set_xlabel('Time (seconds)', fontsize=10)
                ax.set_title(f'Neuron {neuron_idx} - Membrane Potential', fontsize=11)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Spikes
                ax = axes[i, 1]
                ax.plot(time, hidden1_spikes[:, neuron_idx], color=PROF_COLORS['hidden1'], 
                       linewidth=1.5, alpha=0.8, marker='o', markersize=3)
                ax.set_ylabel(f'Neuron {neuron_idx}\nSpikes', fontsize=10, fontweight='bold')
                if i == num_neurons_this_fig - 1:
                    ax.set_xlabel('Time (seconds)', fontsize=10)
                ax.set_title(f'Neuron {neuron_idx} - Spike Output', fontsize=11)
                ax.set_ylim([-0.1, 1.1])
                ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            plt.show()
        
        # ========== FIGURE 3: HIDDEN LAYER 2 NEURONS (if exists) ==========
        if num_layers >= 2 and hidden2_spikes is not None:
            num_hidden2 = hidden2_spikes.shape[1]
            num_figs_h2 = int(np.ceil(num_hidden2 / max_neurons_per_fig))
            
            for fig_idx in range(num_figs_h2):
                start_idx = fig_idx * max_neurons_per_fig
                end_idx = min(start_idx + max_neurons_per_fig, num_hidden2)
                num_neurons_this_fig = end_idx - start_idx
                
                fig, axes = plt.subplots(num_neurons_this_fig, 2, figsize=(16, 3*num_neurons_this_fig))
                if num_neurons_this_fig == 1:
                    axes = axes.reshape(1, -1)
                
                fig.suptitle(f'HIDDEN LAYER 2 - Neurons {start_idx} to {end_idx-1}\n'
                            f'Sample {original_sample_idx}', 
                            fontsize=14, fontweight='bold')
                
                for i, neuron_idx in enumerate(range(start_idx, end_idx)):
                    # Membrane potential
                    ax = axes[i, 0]
                    ax.plot(time, hidden2_mem[:, neuron_idx], color=PROF_COLORS['hidden2'], 
                           linewidth=1.5, alpha=0.8)
                    ax.set_ylabel(f'Neuron {neuron_idx}\nMembrane Pot.', fontsize=10, fontweight='bold')
                    if i == num_neurons_this_fig - 1:
                        ax.set_xlabel('Time (seconds)', fontsize=10)
                    ax.set_title(f'Neuron {neuron_idx} - Membrane Potential', fontsize=11)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    # Spikes
                    ax = axes[i, 1]
                    ax.plot(time, hidden2_spikes[:, neuron_idx], color=PROF_COLORS['hidden2'], 
                           linewidth=1.5, alpha=0.8, marker='o', markersize=3)
                    ax.set_ylabel(f'Neuron {neuron_idx}\nSpikes', fontsize=10, fontweight='bold')
                    if i == num_neurons_this_fig - 1:
                        ax.set_xlabel('Time (seconds)', fontsize=10)
                    ax.set_title(f'Neuron {neuron_idx} - Spike Output', fontsize=11)
                    ax.set_ylim([-0.1, 1.1])
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
        
        # ========== FIGURE 4: HIDDEN LAYER 3 NEURONS (if exists) ==========
        if num_layers >= 3 and hidden3_spikes is not None:
            num_hidden3 = hidden3_spikes.shape[1]
            num_figs_h3 = int(np.ceil(num_hidden3 / max_neurons_per_fig))
            
            for fig_idx in range(num_figs_h3):
                start_idx = fig_idx * max_neurons_per_fig
                end_idx = min(start_idx + max_neurons_per_fig, num_hidden3)
                num_neurons_this_fig = end_idx - start_idx
                
                fig, axes = plt.subplots(num_neurons_this_fig, 2, figsize=(16, 3*num_neurons_this_fig))
                if num_neurons_this_fig == 1:
                    axes = axes.reshape(1, -1)
                
                fig.suptitle(f'HIDDEN LAYER 3 - Neurons {start_idx} to {end_idx-1}\n'
                            f'Sample {original_sample_idx}', 
                            fontsize=14, fontweight='bold')
                
                for i, neuron_idx in enumerate(range(start_idx, end_idx)):
                    # Membrane potential
                    ax = axes[i, 0]
                    ax.plot(time, hidden3_mem[:, neuron_idx], color=PROF_COLORS['hidden3'], 
                           linewidth=1.5, alpha=0.8)
                    ax.set_ylabel(f'Neuron {neuron_idx}\nMembrane Pot.', fontsize=10, fontweight='bold')
                    if i == num_neurons_this_fig - 1:
                        ax.set_xlabel('Time (seconds)', fontsize=10)
                    ax.set_title(f'Neuron {neuron_idx} - Membrane Potential', fontsize=11)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    # Spikes
                    ax = axes[i, 1]
                    ax.plot(time, hidden3_spikes[:, neuron_idx], color=PROF_COLORS['hidden3'], 
                           linewidth=1.5, alpha=0.8, marker='o', markersize=3)
                    ax.set_ylabel(f'Neuron {neuron_idx}\nSpikes', fontsize=10, fontweight='bold')
                    if i == num_neurons_this_fig - 1:
                        ax.set_xlabel('Time (seconds)', fontsize=10)
                    ax.set_title(f'Neuron {neuron_idx} - Spike Output', fontsize=11)
                    ax.set_ylim([-0.1, 1.1])
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
        
        # ========== FIGURE 5: OUTPUT LAYER NEURONS (5 classes) ==========
        fig5, axes5 = plt.subplots(4, 2, figsize=(16, 12))
        fig5.suptitle(f'OUTPUT LAYER - All 4 Classes\n'
                      f'Sample {original_sample_idx} | True Label: {true_risk} (Class {sample_label_val})', 
                      fontsize=14, fontweight='bold')
        
        spike_counts = np.sum(output_spikes, axis=0)
        predicted_class = np.argmax(spike_counts)
        
        for class_idx in range(4):
            # Membrane potential
            ax = axes5[class_idx, 0]
            ax.plot(time, output_mem[:, class_idx], color=CLASS_COLORS[class_idx], 
                   linewidth=2, alpha=0.8, label=f'Class {class_idx}')
            if class_idx == predicted_class:
                ax.axhline(y=0, color=CLASS_COLORS[class_idx], linestyle='--', alpha=0.5, label='PREDICTED')
            if class_idx == sample_label_val:
                ax.axhline(y=0, color='#2C2C2C', linestyle=':', alpha=0.5, label='TRUE LABEL')
            ax.set_ylabel(f'{class_names[class_idx]}\nMembrane Pot.', fontsize=10, fontweight='bold')
            if class_idx == 3:
                ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_title(f'Output Neuron {class_idx} ({class_names[class_idx]}) - Membrane Potential', fontsize=11)
            ax.legend(loc='upper right', fontsize=8, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Spikes
            ax = axes5[class_idx, 1]
            ax.plot(time, output_spikes[:, class_idx], color=CLASS_COLORS[class_idx], 
                   linewidth=2, alpha=0.8, marker='o', markersize=4, label=f'Class {class_idx}')
            if class_idx == predicted_class:
                ax.axhline(y=0.5, color=CLASS_COLORS[class_idx], linestyle='--', alpha=0.5, label='PREDICTED')
            if class_idx == sample_label_val:
                ax.axhline(y=0.5, color='#2C2C2C', linestyle=':', alpha=0.5, label='TRUE LABEL')
            ax.set_ylabel(f'{class_names[class_idx]}\nSpikes', fontsize=10, fontweight='bold')
            if class_idx == 3:
                ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_title(f'Output Neuron {class_idx} ({class_names[class_idx]}) - Spike Output\n'
                        f'Total: {spike_counts[class_idx]:.0f} spikes', fontsize=11)
            ax.set_ylim([-0.1, 1.1])
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n{'='*60}")
        print("SIGNAL FLOW SUMMARY (Individual Neurons)")
        print(f"{'='*60}")
        print(f"Sample: {original_sample_idx}")
        print(f"fPSA Frequency: {f1:.4f} Hz")
        print(f"tPSA Frequency: {f2:.4f} Hz")
        print(f"True Label: {true_risk} (Class {sample_label_val})")
        print(f"\nNetwork Structure:")
        print(f"  Hidden Layer 1: {num_hidden1} neurons")
        if num_layers >= 2:
            print(f"  Hidden Layer 2: {hidden2_spikes.shape[1]} neurons")
        if num_layers >= 3:
            print(f"  Hidden Layer 3: {hidden3_spikes.shape[1]} neurons")
        print(f"  Output Layer: 4 neurons (classes)")
        print(f"\nOutput Spike Counts:")
        for i, count in enumerate(spike_counts):
            marker = " ← PREDICTED" if i == predicted_class else ""
            true_marker = " ← TRUE" if i == sample_label_val else ""
            print(f"  Class {i} ({class_names[i]:18s}): {count:6.0f} spikes{marker}{true_marker}")
        print(f"\nPredicted Class: {class_names[predicted_class]} (Class {predicted_class})")
        print(f"{'='*60}")

    def save_model(self, filepath):
        """
        Save the trained model, optimizer state, training history, and metadata to a file.
        
        Args:
            filepath: Path where to save the model (e.g., 'model.pth' or 'model.pt')
        """
        if not self.training_loop_complete:
            raise ValueError("Model must be trained first. Please complete training_loop() before saving.")
        
        if not self.final_evaluation_complete:
            raise ValueError("Final evaluation must be completed. Please call final_evaluation() before saving.")
        
        # Determine network architecture parameters
        if isinstance(self.net, CancerNet_1layer):
            num_layers = 1
            num_hidden1 = self.net.fc1.out_features
            num_hidden2 = None
            num_hidden3 = None
        elif isinstance(self.net, CancerNet_2layer):
            num_layers = 2
            num_hidden1 = self.net.fc1.out_features
            num_hidden2 = self.net.fc2.out_features
            num_hidden3 = None
        elif isinstance(self.net, CancerNet_3layer):
            num_layers = 3
            num_hidden1 = self.net.fc1.out_features
            num_hidden2 = self.net.fc2.out_features
            num_hidden3 = self.net.fc3.out_features
        else:
            raise ValueError("Unknown network architecture")
        
        # Get beta value from the network
        beta = self.net.lif1.beta.item() if hasattr(self.net.lif1.beta, 'item') else float(self.net.lif1.beta)
        
        # Prepare save dictionary
        save_dict = {
            # Model state
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.create_optimizer_complete else None,
            
            # Network architecture
            'num_layers': num_layers,
            'num_hidden1': num_hidden1,
            'num_hidden2': num_hidden2,
            'num_hidden3': num_hidden3,
            'beta': beta,
            
            # Training history
            'loss_hist': self.loss_hist,
            'test_loss_hist': self.test_loss_hist,
            'train_acc_hist': self.train_acc_hist,
            'test_acc_hist': self.test_acc_hist,
            'num_epochs': self.num_epochs,
            
            # Final evaluation results
            'final_train_acc': self.final_train_acc,
            'final_test_acc': self.final_test_acc,
            
            # Data preprocessing parameters
            'dt': self.dt,
            'T': self.T,
            'a': self.a,
            'target_length': self.target_length,
            'average_temporal_pooling_downsample_signal_complete': self.average_temporal_pooling_downsample_signal_complete,
            
            # Flags
            'create_network_complete': self.create_network_complete,
            'create_optimizer_complete': self.create_optimizer_complete,
            'create_loss_function_complete': self.create_loss_function_complete,
            'training_loop_complete': self.training_loop_complete,
            'final_evaluation_complete': self.final_evaluation_complete,
            'create_data_loaders_complete': self.create_data_loaders_complete,
            'split_dataset_complete': self.split_dataset_complete,
            'create_dataset_complete': self.create_dataset_complete,
            'convert_to_torch_tensor_complete': self.convert_to_torch_tensor_complete,
            'labeling_y_complete': self.labeling_y_complete,
            'generate_synaptic_signals_complete': self.generate_synaptic_signals_complete,
            
            # Batch size for data loaders
            'batch_size': self.batch_size,
        }
        
        # Save to file
        torch.save(save_dict, filepath)
        
        print(f"\n{'='*60}")
        print("MODEL SAVED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Saved to: {filepath}")
        print(f"Network architecture: {num_layers} layers")
        print(f"  Hidden layer 1: {num_hidden1} neurons")
        if num_hidden2 is not None:
            print(f"  Hidden layer 2: {num_hidden2} neurons")
        if num_hidden3 is not None:
            print(f"  Hidden layer 3: {num_hidden3} neurons")
        print(f"Final test accuracy: {self.final_test_acc:.2f}%")
        print(f"{'='*60}")

    def load_model(self, filepath):
        """
        Load a saved model, optimizer state, training history, and metadata from a file.
        After loading, you can call visualization functions like:
        - plot_loss_curves()
        - plot_confusion_matrix()
        - test_single_prediction()
        - visualize_signal_flow2()
        
        Args:
            filepath: Path to the saved model file (e.g., 'model.pth' or 'model.pt')
        """
        if not self.create_data_loaders_complete:
            raise ValueError("Data loaders must be created first. Please call create_data_loaders() before loading model.")
        
        if not self.split_dataset_complete:
            raise ValueError("Dataset must be split first. Please call split_dataset() before loading model.")
        
        if not self.create_dataset_complete:
            raise ValueError("Dataset must be created first. Please call create_dataset() before loading model.")
        
        if not self.convert_to_torch_tensor_complete:
            raise ValueError("Data must be converted to torch tensors first. Please call convert_to_torch_tensor() before loading model.")
        
        if not self.labeling_y_complete:
            raise ValueError("Labels must be labeled first. Please call labeling_y() before loading model.")
        
        if not self.generate_synaptic_signals_complete:
            raise ValueError("Synaptic signals must be generated first. Please call generate_synaptic_signals() before loading model.")
        
        # Load the saved dictionary
        print(f"\n{'='*60}")
        print("LOADING MODEL")
        print(f"{'='*60}")
        print(f"Loading from: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Extract network architecture parameters
        num_layers = checkpoint['num_layers']
        num_hidden1 = checkpoint['num_hidden1']
        num_hidden2 = checkpoint.get('num_hidden2', None)
        num_hidden3 = checkpoint.get('num_hidden3', None)
        beta = checkpoint['beta']
        
        # Recreate the network with the same architecture
        if num_layers == 1:
            self.net = CancerNet_1layer(num_inputs=2, num_hidden1=num_hidden1, num_outputs=4, beta=beta)
        elif num_layers == 2:
            self.net = CancerNet_2layer(num_inputs=2, num_hidden1=num_hidden1, num_hidden2=num_hidden2, num_outputs=4, beta=beta)
        elif num_layers == 3:
            self.net = CancerNet_3layer(num_inputs=2, num_hidden1=num_hidden1, num_hidden2=num_hidden2, 
                                       num_hidden3=num_hidden3, num_outputs=4, beta=beta)
        else:
            raise ValueError(f"Invalid number of layers: {num_layers}")
        
        # Load model state
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()  # Set to evaluation mode
        
        # Load optimizer state if available (optional for visualization)
        if checkpoint.get('optimizer_state_dict') is not None:
            if self.create_optimizer_complete:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    print("Warning: Could not load optimizer state.")
            else:
                print("Note: Optimizer not created yet. Not needed for visualization functions.")
        
        # Load training history
        self.loss_hist = checkpoint['loss_hist']
        self.test_loss_hist = checkpoint['test_loss_hist']
        self.train_acc_hist = checkpoint['train_acc_hist']
        self.test_acc_hist = checkpoint['test_acc_hist']
        self.num_epochs = checkpoint['num_epochs']
        
        # Load final evaluation results
        self.final_train_acc = checkpoint['final_train_acc']
        self.final_test_acc = checkpoint['final_test_acc']
        
        # Restore flags
        self.create_network_complete = checkpoint.get('create_network_complete', True)
        self.create_optimizer_complete = checkpoint.get('create_optimizer_complete', True)
        self.create_loss_function_complete = checkpoint.get('create_loss_function_complete', True)
        self.training_loop_complete = checkpoint.get('training_loop_complete', True)
        self.final_evaluation_complete = checkpoint.get('final_evaluation_complete', True)
        
        # Verify data preprocessing parameters match
        saved_dt = checkpoint.get('dt', None)
        saved_T = checkpoint.get('T', None)
        saved_a = checkpoint.get('a', None)
        saved_target_length = checkpoint.get('target_length', None)
        
        if saved_dt is not None and saved_dt != self.dt:
            print(f"Warning: dt mismatch. Saved: {saved_dt}, Current: {self.dt}")
        if saved_T is not None and saved_T != self.T:
            print(f"Warning: T mismatch. Saved: {saved_T}, Current: {self.T}")
        if saved_a is not None and saved_a != self.a:
            print(f"Warning: a mismatch. Saved: {saved_a}, Current: {self.a}")
        if saved_target_length is not None and saved_target_length != self.target_length:
            print(f"Warning: target_length mismatch. Saved: {saved_target_length}, Current: {self.target_length}")
        
        print(f"\nModel loaded successfully!")
        print(f"Network architecture: {num_layers} layers")
        print(f"  Hidden layer 1: {num_hidden1} neurons")
        if num_hidden2 is not None:
            print(f"  Hidden layer 2: {num_hidden2} neurons")
        if num_hidden3 is not None:
            print(f"  Hidden layer 3: {num_hidden3} neurons")
        print(f"Beta: {beta}")
        print(f"Training epochs: {self.num_epochs}")
        print(f"Final train accuracy: {self.final_train_acc:.2f}%")
        print(f"Final test accuracy: {self.final_test_acc:.2f}%")
        print(f"\nYou can now call:")
        print(f"  - plot_loss_curves()")
        print(f"  - plot_confusion_matrix()")
        print(f"  - test_single_prediction()")
        print(f"  - visualize_signal_flow2()")
        print(f"{'='*60}")

    def clear_all(self):

        print(f"\n{'='*60}")
        print("CLEARING ALL DATA")
        print(f"{'='*60}")

        self.X_fPSA = []
        self.X_tPSA = []
        self.y = []
        self.N = 0
        self.dt= 0
        self.T= 0
        self.a= 0
        self.target_length=0

        self.X_tPSA_downsampled = None
        self.X_fPSA_downsampled = None

        self.y_numeric = None
        self.data_combined = None
        self.y_torch = None
        self.full_dataset = None

        self.train_dataset = None
        self.test_dataset = None

        self.batch_size = 0
        self.train_loader = None
        self.test_loader = None

        self.net = None

        self.all_preds = None
        self.all_labels = None

        self.final_train_acc = 0
        self.final_test_acc = 0

        self.loss_hist = None
        self.test_loss_hist = None
        self.train_acc_hist = None
        self.test_acc_hist = None


        #-----Flags-----
        self.generate_synaptic_signals_complete = False
        self.labeling_y_complete = False
        self.average_temporal_pooling_downsample_signal_complete = False
        self.convert_to_torch_tensor_complete = False
        self.create_dataset_complete = False
        self.split_dataset_complete = False
        self.create_data_loaders_complete = False
        self.create_network_complete = False
        self.create_loss_function_complete = False
        self.create_optimizer_complete = False
        self.training_loop_complete = False
        self.final_evaluation_complete = False




    def run_full_pipeline(self,dt = 1e-3 , T = 20,a = 0.9,target_length=25,downsampled_signals=True,
        split_ratio=0.8,batch_size=32,num_epochs=400,num_layers=3,num_hidden1=60,num_hidden2=20,num_hidden3=10,
        beta=0.9,correct_rate=0.8,incorrect_rate=0.2,lr=1e-3,betas=(0.9, 0.999),generation_method='1',save_model=False,model_name='SNN_Model'):


        self.generate_synaptic_signals(dt = dt , T = T,a = a,generation_method=generation_method)
        self.labeling_y()
        self.plot_synaptic_signals()
        if downsampled_signals==True:
            self.average_temporal_pooling_downsample_signal(target_length=target_length)
            self.plot_comparison_downsampled_signals()
        self.convert_to_torch_tensor(downsampled_signals=downsampled_signals)
        self.create_dataset()
        self.split_dataset(split_ratio=split_ratio)
        self.create_data_loaders(batch_size=batch_size)
        self.create_network(num_layers=num_layers,num_hidden1=num_hidden1,num_hidden2=num_hidden2,num_hidden3=num_hidden3,beta=beta)
        self.create_loss_function(correct_rate=correct_rate,incorrect_rate=incorrect_rate)
        self.create_optimizer(lr=lr,betas=betas)
        self.training_loop(num_epochs=num_epochs)
        self.final_evaluation()
        self.plot_loss_curves()
        self.plot_confusion_matrix()
        self.test_single_prediction()

        if save_model==True:
            self.save_model(filepath=f'{model_name}.pth')
        
