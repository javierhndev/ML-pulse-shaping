# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:47:47 2023

@author: Sean Buczek
"""

import numpy as np

def find_closest(A, target):
    ''' 
    Compares two arrays to find the what values are closest between them. 
    Returned array is the shape of target where each member is the index of A 
    that is closest to the value of target at that position.
    
    Example:
        A = [1,2,3,4,5]
        Target = 5*np.random.rand(6) = [1.56965366, 0.89292633, 4.69703774, 
                                        2.90827883, 1.93193723,2.62520874]
        find_closest(A,target) = array([1, 0, 4, 2, 1, 2], dtype=int64)
    '''
    if not isinstance(target,np.ndarray): target = np.array([target])
    if A[0]-A[1]<0:
        idx = A.searchsorted(target)
        idx = np.clip(idx, 1, len(A)-1)
        left = A[idx-1]
        right = A[idx]
        idx -= target - left < right - target
    elif A[0]-A[1]>0:
        B = np.flip(A)
        idy = B.searchsorted(target)
        idy = np.clip(idy, 1, len(B)-1)
        left = B[idy-1]
        right = B[idy]
        idy -= target - left < right - target
        idx = np.zeros(len(idy))
        for i,x in enumerate(idy):
            test = B[x]
            idx[i] = np.where(A==test)[0][0]
        idx = int(idx)
    return idx
    
class pulse:
    '''
    A class for modeling pulses based on their spectrum and phase.
    
    Parameters:
        time: (int or array) Either a float defining the largest time (in 
            femtoseconds) from the center of the pulse to calculate (ex: 100 
            means your time window goes from -100fs to 100fs) or a fully 
            defined time array. Unit: femtoseconds
            
        spectralRange: (array) An array of either the wavelengths used in the 
            spectrum data or the frequencies used. Define which using the 
            wavelength variable. Unit: meters or hertz
            
        spectralIntensity: (array) Array of intensities for the spectrum. 
            It should be the same length and in the same order order as the 
            array used for the spectralRange variable. Unit: arb.
            
        phase: (array or None) Array of phase values for the spectrum. It 
            should be the same length and in the same order as the array used 
            for the spectralRange variable. If None (default), an array of all
            zeros of the same length as spectralRange will be generated.
            Unit: radians
            
        center: (float or None) Value of the central wavelength/frequency of 
            the spectrum. Use of wavelength or frequency should be the same as
            the spectralRange array. If None (default), central frequency will 
            be calculated as the median value of spectralRange array.
            Unit: meters or hertz
            
        wavelength: (Boolean) True means you provided wavelengths for 
            spectralRange, False means you provided frequencies. Default: True
    
    '''
    def __init__(self,time,spectralRange,spectralIntensity,phase=None,center=None,wavelength=True):
        if isinstance(time,int):
            self._time = np.linspace(-time,time,20*time+1)*1e-15
        elif isinstance(time,np.ndarray) or isinstance(time,list):
            self._time = time*1e-15
        else:
            raise ValueError('Please provide either a time array or a real valued time boundary.')
        if wavelength:
            self._wavelength = spectralRange.copy()
            self._frequency = 299792458/spectralRange.copy()
            if center is not None: 
                self._center = find_closest(self._wavelength,center)
        else:
            self._frequency = spectralRange.copy()
            self._wavelength = 299792458/spectralRange.copy()
            if center is not None: 
                self._center = find_closest(self._frequency,center)
        if center is None: self._center = len(self._frequency)//2
        if isinstance(phase,np.ndarray):
            self._specphase = phase.copy()
            self._absphase = phase.copy()
        elif phase == None:
            self._specphase = np.zeros(len(spectralRange))
            self._absphase = np.zeros(len(spectralRange))
        else:
            raise ValueError('Please provide either a phase array or None (for a perfectly FTL pulse)')
        self._specint = spectralIntensity.copy()
        self.field = True
    
    @property
    def field(self):
        ''' 
        Returns the normalized field for the pulse based on the input spectrum
        and phase.
        '''
        return self._field.copy()
    
    @field.setter
    def field(self,update):
        '''
        Generates the normalized field for the pulse based on the input 
        spectrum and phase.
        '''
        if update:
            wave = np.zeros([len(self._frequency),len(self._time)],dtype=complex)
            ew = np.zeros(len(self._frequency),dtype=complex)
            for i in range(len(self._frequency)):
                ew[i] = np.sqrt(self._specint[i])*np.exp(-1j*(self._specphase[i]-self._specphase[self._center]))    
            for i in range(len(self._time)):
                wave[:,i] = ew*np.exp(2j*np.pi*self._frequency*self._time[i])
            f = np.sum(wave,axis=0)
            self._field = f/np.max(f)
    
    @property
    def centerwl(self):
        ''' 
        Returns the center wavelength used in calculations.
        
        Read only access.
        '''
        return self._wavelength[self._center][0]
    
    @property
    def centerfreq(self):
        ''' 
        Returns the center frequency used in calculations.
        
        Read only access.
        '''
        return self._frequency[self._center][0]
    
    @property
    def centerfield(self):
        ''' 
        Returns the normalized waveform of the center wavelength.
        
        Read only access.
        '''
        return np.exp(-2j*np.pi*self._frequency[self._center]*self._time)
    
    @property
    def wavelength(self):
        ''' 
        Returns the wavelength array used to calculate the pulse.
        
        Read only access.
        '''
        return self._wavelength.copy()
    
    @property
    def wlnano(self):
        ''' 
        Returns the wavelength array used to calculated the pulse in units of nanometers.
        
        Read only access
        '''
        return self._wavelength.copy()*1e9
    
    @property
    def frequency(self):
        ''' 
        Returns the frequency array used to calculate the pulse.
        
        Read only access.
        '''
        return self._frequency.copy()
    
    @property
    def time(self):
        ''' 
        Returns the time array used to calculate the pulse.
        
        Read only access.
        '''
        return self._time.copy()
    
    @property
    def tfemto(self):
        ''' 
        Returns the time array used to calculate the pulse in units of femtoseconds.
        
        Read only access.
        '''
        return self._time.copy()/1e-15
    
    @property
    def intensity(self):
        ''' 
        Returns the normalized time domain intensity of the pulse.
        
        Read only access.
        '''
        return np.square(abs(self.field.copy()))/np.max(np.square(abs(self.field)))
    
    @property
    def duration(self):
        ''' 
        Returns the fwhm duration the pulse.
        '''
        half = (max(self.intensity.copy())-min(self.intensity.copy()))/2+min(self.intensity.copy())
        low = np.where(self.intensity.copy()>half)[0][0]
        high = np.where(self.intensity.copy()>half)[0][-1]
        return abs(self._time.copy()[high]-self._time.copy()[low])
    
    @property
    def sphase(self):
        ''' 
        Returns the spectral phase array used to calculate the pulse.
        '''
        return self._specphase.copy()
    
    @sphase.setter
    def sphase(self,phase):
        ''' 
        Sets the spectral phase of the pulse to be the provided phase values.
        Overwrites the existing phase array of the pulse.
        Recalculates the pulse field after.
        '''
        if isinstance(phase,np.ndarray) and len(phase) == len(self._specphase): self._specphase = phase.copy()
        elif phase is None: self._specphase = np.zeros(len(self._wavelength))
        else: raise ValueError('Please provide a phase array of the same length as existing phase array, or None.') 
        self.field = True
    
    @property
    def tphase(self):
        ''' 
        Returns the temporal phase of the pulse.
        
        Read only access.
        '''
        temp = np.unwrap(-np.imag(np.log(self._field.copy()))+np.imag(np.log(self.centerfield)))
        return temp - temp[len(self._time)//2]
    
    @property
    def instfreq(self):
        ''' 
        Returns the instananeous frequency (Hz) as a function of time.
        
        Read only access.
        '''
        return self._frequency[self._center] - np.gradient(self.tphase,self._time)/(2*np.pi)
    
    @property
    def spectrum(self):
        ''' 
        Returns the spectral intensity array used to calculate the pulse.
        It's a normalized copy of the spectrum passed into the pulse.
        
        Read only access.
        '''
        return self._specint.copy()/np.max(self._specint)
    
    @property
    def specwidth(self):
        '''
        Returns the FWHM width of the spectrum in meters.
        
        Read only access.

        '''
        half = (max(self._specint.copy())-min(self._specint.copy()))/2+min(self._specint.copy())
        low = np.where(self._specint.copy()>half)[0][0]
        high = np.where(self._specint.copy()>half)[0][-1]
        return abs(self._wavelength.copy()[high]-self._wavelength.copy()[low])
    
    @property
    def ftlduration(self):
        waveftl = np.zeros([len(self._frequency),len(self._time)],dtype=complex)
        for i in range(len(self._frequency)):
            waveftl[i] = np.sqrt(self._specint[i])*np.exp(-2j*np.pi*self._frequency[i]*self._time)
        ftl = np.sum(waveftl,axis=0)  
        ftl *=1/np.max(ftl)
        ftlint = np.square(abs(ftl))/np.max(np.square(abs(ftl)))
        half = (max(ftlint)-min(ftlint))/2+min(ftlint)
        low = np.where(ftlint>half)[0][0]
        high = np.where(ftlint>half)[0][-1]
        return abs(self._time.copy()[high]-self._time.copy()[low])
    
    def dispersion(self,order,amount):
        ''' 
        Adds dispersion to the phase of the pulse.
        Type of dispersion is determined by the order (ex: 3 for Third Order Dispersion), 
        with amount determining the amount of that dispersion order (ex: 5000 would mean 5000 fs^3 for TOD).
        
        The pulse field is then recalculated.
        '''
        if not isinstance(order,int): 
            raise ValueError('Please provide an integer greater than or equal to zero for the dispersion order.')
        # when1 = np.where(self.intensity == self.intensity.max())[0][0]
        disp = np.zeros(len(self._frequency))
        amount = amount*(1e-15)**order
        for i,fr in enumerate(self._frequency.copy()):
            disp[i] = (amount*(2*np.pi*(fr-self.centerfreq))**order)/(np.math.factorial(order))
        self.sphase = self.sphase + disp
        # when2 = np.where(self.intensity == self.intensity.max())[0][0]
        # delay = (when2-when1)*abs(self._time[0]-self._time[1])
        # for i,fr in enumerate(self._frequency.copy()):
        #     disp[i] = delay*2*np.pi*(fr-self.centerfreq)
        # self.sphase = self.sphase + disp
        ####possible improvement: automatically calculating delay needed:
            #### Calculate when the peak of the original pulse is
            #### Calculate when the peak of the new pulse is after dispersion added
            #### Find difference
            #### Automatically add that difference as a 1st order dispersion