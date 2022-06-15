from math import pi

def calculate_freq_and_cap(meas, L, Cfilt, gain=4.0): # gain is no. of bits shifted
  freqs = [40.0e6 * mea / (2**(12+gain)) for mea in meas]
  caps  = [1.0 / (L*(2.0*pi*freq)**2) - Cfilt for freq in freqs]
  return freqs, caps

def calculate_drag_force(v1, v2, v3, v4):
  X = -v1 -v2 +v3 +v4
  drag = 15695275 * X + 9088 # lbf
  drag_kn = 4.44822 * drag
  return drag_kn