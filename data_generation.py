# data_generation.py
"""
**NOTE**: Only need to run this script if you have the raw data from `MORIA` and want
 to reconstuct the data used in the modeling. Otherwise just simply use the `hdf5`
   stored in this repository.

In any experimental run in `GALADRIEL`, thousands of shots are generated and several
 diagnostics/tools measure the parameters and results. The data is stored in
   `MORIA` (a MongoDB database).

The data required for the ML modeling in this repository is:
- Input Dazzler coefficients (and goodness)
- Pulse shape

Dazzler coefficients used in the experimtal run can easily been extracted from `Moria`.
However pulse reconstruction data directly from `Wizzler` has a very low 
resolution (~20 points). To overcome this issue, we can reconstruct the pulse using
 the method described in ["High-resolution direct phase control in the spectral domain
   in ultrashort pulse lasers for pulse-shaping applications"](https://doi.org/10.1088/1748-0221/20/05/P05002).
     In this script we use that algorithm to reconstruct the pulse shape with the
       desired resolution (in our case, 200 points). To run it we need the frequency,
         intensity and phase captured by the Wizzler.

This script assumes that the data (Dazzler, wizzler...) is a in a `hdf5` file
 previously pulled from `MORIA` . And it returns a new `hdf5` file winth only the Dazzler coefficients and pulse shapes from the experiment. This notebook only needs to be executed once. This way that data can be used for model training/testing without recalculating pulse shapes.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulse

# READ DATASET
# READ DATASET (HR) FROM Galadriel database
filename = '../laps-ml/datasets/galadriel_dataset_24_09_18_high_res.h5'
#filename = '../laps-ml/datasets/galadriel_dataset_25_03_13_high_res.h5'

print('Reading input data from:', filename)

# read the Dazller input data
df_input = pd.read_hdf(filename, 'df_input')

# Extract values from spec
df_spec_freq = pd.read_hdf(filename, 'df_spec_freq')
df_spec_int = pd.read_hdf(filename, 'df_spec_int')
df_spec_phase = pd.read_hdf(filename, 'df_spec_phase')

# Extract time and intensity (calculated by Wizzler) for comparison
df_time = pd.read_hdf(filename, 'df_time')
df_time_val = df_time.loc[0]
df_pulse_wizz = df_time.drop([0])  # drop the time values
df_pulse_wizz.reset_index(inplace=True, drop=True)

# Pulse reconstruction function
def pulse_reconst(df_spec_int, df_spec_freq, df_spec_phase, time=100, time_bins=100):
    t = np.linspace(-time, time, time_bins + 1)  # time array, in fs.
    c = 299792458  # speed of light in m/s
    center_freq = c / 800e-9  # center frequency using 800nm
    p_list = []
    for i in range(len(df_spec_freq)):
        intensity = df_spec_int.loc[i].to_numpy()
        frequency = df_spec_freq.loc[i].to_numpy()
        phase = df_spec_phase.loc[i].to_numpy()
        p = pulse.pulse(t, frequency * 1e12, intensity, phase, wavelength=False, center=center_freq)
        p_list.append(p.intensity)
    return t, pd.DataFrame(p_list)

# RECONSTRUCT PULSES
write_data =  False # set to True to write the reconstructed pulse data to an hdf5 file
output_filename = 'datasets/pulse_and_dazzler_240918.h5'
if write_data:
    print('Reconstructing pulses and writing to:', output_filename)
    t_200, df_pulse_200 = pulse_reconst(df_spec_int, df_spec_freq, df_spec_phase, time=150, time_bins=200)
    # write to hdf5 file
    pd.DataFrame(t_200).to_hdf(output_filename, key='df_time_200', mode='w')
    df_pulse_200.to_hdf(output_filename, key='df_pulse_200')
    df_input.to_hdf(output_filename, key='df_input')
else:
    print('Skipping pulse reconstruction and writing to file. Set write_data to True to enable this step.')




# Save raw data from a pulse close to FTL (o2=35000, o3=0, o4=-4.5e6), e.g. 168758
print(' ')
print('Saving raw data for a shot close to FTL to an hdf5 file.')
shot_ftl = 168758
ref_index = df_input[df_input['shot number'] == shot_ftl].index.values[0]
intensity = df_spec_int.loc[ref_index]
frequency = df_spec_freq.loc[ref_index]
phase = df_spec_phase.loc[ref_index]
output_ftl_filename = 'datasets/raw_pulse_ftl_168758_240918.h5'
# save the raw data for this shot to an hdf5 file
intensity.to_hdf(output_ftl_filename, key='intensity')
frequency.to_hdf(output_ftl_filename, key='frequency')
phase.to_hdf(output_ftl_filename, key='phase')
df_input[df_input['shot number'] == shot_ftl].to_hdf(output_ftl_filename, key='dazzler_input')

# Verify that the files were written correctly by reading the ftl file
df_input_ftl = pd.read_hdf(output_ftl_filename, key='dazzler_input')
print(' ')
print('Data for shot number (FTL)', shot_ftl, 'saved to:', output_ftl_filename)
print(df_input_ftl)

# Testing the results
# READ HDF5 to verify data
df_time_200 = pd.read_hdf(output_filename, key='df_time_200')
df_pulse_200 = pd.read_hdf(output_filename, key='df_pulse_200')
df_input_read = pd.read_hdf(output_filename, key='df_input')
t_200 = df_time_200.to_numpy()


import matplotlib
matplotlib.use('TkAgg')

rand_index = 3
plt.plot(t_200, df_pulse_200.loc[rand_index], label='Reconstructed Pulse')
plt.plot(df_time_val.values, df_pulse_wizz.loc[rand_index], label='Wizzler (low res) Pulse')
plt.xlabel('Time (fs)')
plt.ylabel('Intensity (a.u.)')
plt.text(-120, 0.8, r'$\beta_2$=' + str(df_input_read['order2'].iloc[rand_index]), fontsize=12)
plt.text(-120, 0.7, r'$\beta_3$=' + str(df_input_read['order3'].iloc[rand_index]), fontsize=12)
plt.text(-120, 0.6, r'$\beta_4$=' + str(df_input_read['order4'].iloc[rand_index]), fontsize=12)
plt.legend()
plt.title('Pulse Reconstruction Comparison')
plt.show()
