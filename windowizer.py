import scipy.signal.windows
import numpy as np
import pdb

def window_maker(shape, num_samples):
  """
  Returns a window suitable for use with the windowizer
  """
  window = scipy.signal.windows.get_window(shape, num_samples)
  return np.expand_dims(window, axis=-1)

class Windowizer:
  """
  Object to hold and apply window to data with specified overlap
  """
  def __init__(self, window_array, overlap_ratio):
    self.my_win = window_array
    self.overlap_ratio = overlap_ratio

  def windowize(self, data, labels, flatten=True):
    """
    Return windowed samples as feature vectors and corresponding labels
    Only return windows for which all data has the same labels
    """
    windowed_data = []
    new_labels = []
    step_size = int(len(self.my_win)*(1.0-self.overlap_ratio))
    for index in list(range(0,len(data)-len(self.my_win)+1,step_size)):
      # check for consistant labels over window
      label_check_fail = False
      for label_check_index in list(range(index,index+len(self.my_win))):
        if labels[index] != labels[label_check_index]:
          label_check_fail = True
          break # stop the check, its already failed
      if label_check_fail:
        continue # skip this window location
      glimpse = np.array(data[index:index+len(self.my_win)], dtype=float)
      windowed_datum = np.multiply(self.my_win, glimpse)
      if flatten:
        windowed_datum = windowed_datum.flatten('F')
      windowed_data.append(windowed_datum)
      new_labels.append(labels[index])
    return windowed_data, new_labels

    
  