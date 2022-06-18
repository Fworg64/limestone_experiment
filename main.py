import pdb
import time
import os

from sklearn import svm
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np

from dataloading.loaders import load_strain_gauge_limestone, load_cap_limestone
from windowizer import Windowizer, window_maker
from custom_pipeline_elements import SampleScaler, ChannelScaler, FFTMag

number_parallel_jobs = 6

window_duration = 0.1 # seconds
window_overlap  = 0.5 # ratio of overlap [0,1)
window_shape    = "boxcar" # from scipy.signal.windows

number_cross_validations = 30
my_test_size = 0.25

# Load data
cap_fs = 400 # Samples per second for each channel
lcm_fs = 537.6 # Samples per second for each channel

print("Loading data...")
this_time = time.time()

raw_cap_data, cap_metadata = load_cap_limestone()
raw_lcm_data, lcm_metadata = load_strain_gauge_limestone()

# Apply windowing
cap_window = Windowizer(window_maker(window_shape, int(window_duration*cap_fs)), window_overlap)
lcm_window = Windowizer(window_maker(window_shape, int(window_duration*lcm_fs)), window_overlap)
windowed_cap_data, windowed_cap_labels = cap_window.windowize(raw_cap_data, cap_metadata)
windowed_lcm_data, windowed_lcm_labels = lcm_window.windowize(raw_lcm_data, lcm_metadata)

# set up data for iteration
full_sensor_data_sets = {"cap":(windowed_cap_data, windowed_cap_labels),
                         "sg" :(windowed_lcm_data, windowed_lcm_labels)}

matr_classes2ints = {"Concrete":0, "Limestone":1}
matr_ints2classes = {v: k for k,v in matr_classes2ints.items()}
  
wear_classes2ints = {"New":0, "Mod.":1, "Worn":2}
wear_ints2classes = {v: k for k,v in wear_classes2ints.items()}

classification_features = {"mat.":(matr_classes2ints, matr_ints2classes),
                           "wear":(wear_classes2ints, wear_ints2classes)}

# Seperate and build material and wear datasets using metadata
app_data_sets = {}
for sens, dataset in full_sensor_data_sets.items():
  for feature, translations in classification_features.items():
    data_X = []
    data_Y = []
    for vector, label in zip(dataset[0], dataset[1]):
      if label[feature] != "NonClassified":
        data_X.append(vector) # capture valid samples
        data_Y.append(translations[0][label[feature]]) # get integer labels
    app_data_sets["%s %s" % (sens, feature)] = (np.array(data_X), np.array(data_Y))
      
# Build preprocessing lists for pipeline
# scale1: [std, samp, chan, none]
# freq_transform: [abs(rfft()), abs(rfft()).^2, sqrt(abs(rfft())), none]
# scale2: [std, samp, chan, none]

that_time = time.time()
print("Data loaded in {0} sec; performing experiments".format(that_time - this_time),
      end='', flush=True)
this_time = time.time()
# Build pipeline
scalings1 = [("FeatureScaler1", StandardScaler()), ("ScaleControl1", None)]
scalings2 = [("FeatureScaler2", StandardScaler()), ("ScaleControl2", None)]
freq_transforms = [('FFT_Mag', FFTMag(4)), ('FFT_MagSq', FFTMag(4,"SQUARE")),
                   ('FFT_MagRt', FFTMag(4,"SQRT")), ("FreqControl", None)]

#pdb.set_trace()
# Do experiment, record data to list
results = [["application", "num_splits", "num_samples", "test_ratio",
            "window_duration", "window_overlap", "window_shape"
            "stand1", "fft", "stand2", "mean_score", "std_dev"]]
for name, (data_X, data_Y) in app_data_sets.items():
 for ft in freq_transforms:
  for sc1 in scalings1:
   for sc2 in scalings2:
     cross_val = ShuffleSplit(n_splits=number_cross_validations, test_size=my_test_size, 
                              random_state = 711711)
     my_pipeline = Pipeline([sc1, ft, sc2, ('svc', svm.SVC(class_weight='balanced'))])
     scores = cross_val_score(my_pipeline, data_X, data_Y, cv=cross_val,
                               scoring='f1_macro', n_jobs=number_parallel_jobs)
     results.append([name, str(number_cross_validations), str(data_X.shape[0]), str(my_test_size),
                     str(window_duration), str(window_overlap), window_shape,
                     my_pipeline.steps[0][0], my_pipeline.steps[1][0], my_pipeline.steps[2][0],
                     str(scores.mean()), str(scores.std())])
     print(".", end='', flush=True)

that_time = time.time()
print(" Done! Took {0} sec; Saving data...".format(that_time - this_time))
# print list and save to file
for result in results:
  print(result)

#pdb.set_trace()
os.makedirs('./out', exist_ok=True)
timestr = time.strftime("%Y%m%d_%H%M%Sresults.csv")
with open('./out/' + timestr, 'w') as f:
  for line in results:
    f.write(','.join(line) + '\n')

print("Have a nice day!")

# Score pipelines using default SVM with linear kernal
# Iterate through material and wear for both sensors
# and all hyperparams using the chosen window settings
# also test different test train splits
