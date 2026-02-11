'''
This script reads galadriel dataset and writes some of its data to an HDF5 file
The output dataset (galadriel_dataset.h5) contains:
    -df_input: DataFrame with the shot number, order2, order3, and order4
    -df_time: Dataframe where the first row is the time array and the followings rows are the pulse shapes
    -df_freq_int: Dataframe where the first row are frequency values and the followings are the intensity values from raw data

The script skips the duplicate shots and the shots with no wizzler data

It also calculates the 'goodness' (ratio between the input and output spectral width)

In its current form, it is the one used to create the 240918 galadriel dataset with goodness

This script requires galadriel_db package to run. Not available in this repository.

Created by Javier Hernandez on Sep 2024. Based on Austin and Gil scripts
'''

#using wiz_data (query_data['results']['wizzler'][j]['data']) get the width using
def get_input_fwhm(wiz_data,ratfrac):
    wizzspect_freq = np.array(wiz_data['spec_data']['frequencies'])
    wizzinput = np.array(wiz_data['spec_data']['intensity'])
    wizzspect_wavelength = 1/(wizzspect_freq*1000/3E8)
    wizzinputmax = np.max(wizzinput)
    wizzinputmaxloc = np.argmax(wizzinput)
    bottomwizzwav = wizzspect_wavelength[np.argmin(np.abs(wizzinput[0:wizzinputmaxloc] - wizzinputmax*ratfrac))]
    topwizzwav = wizzspect_wavelength[wizzinputmaxloc+1 + np.argmin(np.abs(wizzinput[wizzinputmaxloc+1:-1] - wizzinputmax*ratfrac))]
    inputfwhm = np.abs(bottomwizzwav - topwizzwav)

    return inputfwhm

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_raw_fwhm(wiz_data,ratfrac):
    rawspect_wavelength = np.array(wiz_data['raw_data']['wavelengths'])
    rawinput = np.array(wiz_data['raw_data']['instensity'])
    rawinput_sm = moving_average(rawinput, n=15)
    rawinputmax = np.max(rawinput_sm)
    rawinputmaxloc = np.argmax(rawinput_sm)
    bottomrawwav = rawspect_wavelength[np.argmin(np.abs(rawinput_sm[0:rawinputmaxloc] - rawinputmax*ratfrac))]
    toprawwav = rawspect_wavelength[rawinputmaxloc+1 + np.argmin(np.abs(rawinput_sm[rawinputmaxloc+1:-1] - rawinputmax*ratfrac))]
    rawfwhm = np.abs(bottomrawwav - toprawwav)
    return rawfwhm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import galadriel_db
gdb = galadriel_db.GaladrielDatabase(1)

#write the dataset to file:
filename='galadriel_dataset_24_09_18.h5'

# build query
shot_start, shot_end = 163165,172668 
gdb.query_diagnostic('ML_SDSC')
gdb.query_shot_range(shot_start, shot_end)
query_data = gdb.run_query()

ratfrac = 0.1 #This guy is arbitrary, but matches the 'Goodness Bar' best as opposed to 0.5 (FWHM) or 0.25 (QWHM)

daz = query_data['results']['dazzler']
wiz = query_data['results']['wizzler']

#Define the lists where we store the dataset
input_list=[]
time_list=[]
freq_int_list=[]

i, j, cur_shot = 0, 0, 0
first_save=0
no_data=0
num_shots = (shot_end - shot_start) + 1
print('')
print('Main loop starts')
while (cur_shot < shot_end):
    daz_data = daz[i]['data']
    daz_meta = daz[i]['metadata']
    cur_shot = daz_meta["shot number"]
 
    #print('')
    wiz_data = wiz[j]['data']
    wiz_meta = wiz[j]['metadata']

    #print(cur_shot)
    #print(query_data['results']['wizzler'][j]['data']['time_data']['time'])
    #print(i,daz_meta["trigger_timestamp"], daz_meta["shot number"], ":", daz_data["delay"], 
    #      daz_data["order2"], daz_data["order3"], daz_data["order4"])
    if wiz_meta["shot number"] == daz_meta["shot number"]: #data OK, save it
        if first_save==0: #only save time and frequency once
            time_list.append(wiz_data['time_data']['time'])
            freq_int_list.append(wiz_data['raw_data']['wavelengths'])
            first_save=1
        #add wizzler data to the lists
        time_list.append(wiz_data['time_data']['intensity'])
        freq_int_list.append(wiz_data['raw_data']['instensity'])

        #input spectrum fwhm
        inputfwhm = get_input_fwhm(wiz_data,ratfrac)
        #raw spectrum FWHM
        rawfwhm = get_raw_fwhm(wiz_data,ratfrac)

        greatness = rawfwhm/inputfwhm
        #print(inputfwhm,rawfwhm,greatness)
        #add dazzler data and the goodness to the list
        input_list.append([int(daz_meta["shot number"]),daz_data["order2"],daz_data["order3"],daz_data["order4"],greatness])
        i += 1
        j += 1

    # duplicate shot numbers
    elif wiz_meta["shot number"] < daz_meta["shot number"]:
        print(f'Duplicate wizzler pulse {wiz_meta["shot number"]}, should be {daz_meta["shot number"]}')
        j += 1
        no_data +=1

    # no data
    else:
        print(f'No wizzler data for pulse {daz_meta["shot number"]}')
        i += 1
        no_data +=1

print(' ')
print(f'We skipped {no_data} shots with no available data')
print(' ')
print('Creating the dataframes')
#create the DataFrame fromt the list
df_input=pd.DataFrame(input_list)
df_input.columns=['shot number','order2','order3','order4','goodness']

#NOTE: First row is the time array
df_time=pd.DataFrame(time_list)

df_freq_int=pd.DataFrame(freq_int_list)



print('Writing the Dataframes to HDF5 files')
#write the Dataframes to a file
df_input.to_hdf(filename,key='df_input',mode='w')
df_time.to_hdf(filename,key='df_time')
df_freq_int.to_hdf(filename,key='df_freq_int')

#read the HDF5 file
print(pd.read_hdf(filename,'df_input'))
print(pd.read_hdf(filename,'df_time'))
print(pd.read_hdf(filename,'df_freq_int'))



