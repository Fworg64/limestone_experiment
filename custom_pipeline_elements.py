#function for the different scalers- chanel and sample


import numpy as np
import pdb

class SampleScaler:
    """
    class for scaling individual sample means and deviations
    """
    def __init__(self):
       pass

    def fit(self, x, y=None, **fit_params):
       return self

    def transform(self, x):
       u = np.expand_dims(np.mean(x,axis = 1), axis = -1)
       s = np.expand_dims(np.std(x,axis = 1), axis = -1)
       z = (x - u) / s
       return z

class ChannelScaler:
    """
    Class for scaling individual channels within samples to zero mean and unit std. dev
    Reduces to sample scaler for num_channels=1
    """
    
    def __init__(self, num_channels=1):
      self.num_channels = num_channels

    def fit(self, x, y=None, **fit_params):
      dims = np.shape(x)
      if dims[1] % self.num_channels != 0:
        raise IndexError("Number of features must be divisable by number of channels!")
      return self
      
    def transform(self, x):
      z = np.array(x)
      z = z.reshape((z.shape[0],self.num_channels,-1), order='F')
      u = np.expand_dims(np.mean(z,axis = -1), axis = -1)
      s = np.expand_dims(np.std(z,axis = -1), axis = -1)
      z = (z - u) / s
      z = z.reshape((z.shape[0], -1), order='F')
      return z

class FFTMag:
    """
    Class for computing magnitude of right side of FFT of input signal,
    optional num_channels parameter causes computation to be broken down along chunks of signal
    optional power parameter transforms data with SQRT or SQUARE, SQUARE is approx PSD
    Input must be 2D np.array with second dimension multiple of number of channels
    First dim is samples, second dim is features of samples
    Transform returns the right side of the FFT magnitude, reducing features roughly by factor of 2
    """

    def __init__(self, num_channels=1, power=None):
      self.num_channels = num_channels
      self.power = power
      self.recognized_powers = {  "SQRT": lambda x : np.sqrt(x), 
                                "SQUARE": lambda x : np.multiply(x,x),
                                    None: lambda x : x}
      if power not in self.recognized_powers:
        raise ValueError("power param must be in %s" % (str(recognized_powers)))

    def fit(self, x, y=None, **fit_params):
      dims = np.shape(x)
      if dims[1] % self.num_channels != 0:
        raise IndexError("Number of features must be divisable by number of channels!")
      return self


    def transform(self, x):
      z = np.array(x)
      z = z.reshape((z.shape[0],self.num_channels,-1), order='F')
      z = np.abs(np.fft.rfft(z, axis=1)).reshape((z.shape[0], -1), order='F')
      z = self.recognized_powers[self.power](z)
      return z

