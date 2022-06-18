import pdb

from sklearn import svm
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np

from dataloading.loaders import load_strain_gauge_limestone, load_cap_limestone
from Windowizer import Windowizer, window_maker

window_duration = 0.1 # seconds
window_overlap  = 0.5 # ratio of overlap [0,1)
window_shape    = "boxcar" # from scipy.signal.windows

number_cross_validations = 3
my_test_size = 0.25

# Load data
cap_fs = 400 # Samples per second for each channel
lcm_fs = 537.6 # Samples per second for each channel
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


# Build pipeline
my_pipeline = Pipeline([('scaler', StandardScaler()), ('svc', svm.SVC())])

# Do experiment
for name, (data_X, data_Y) in app_data_sets.items():
  cross_val = ShuffleSplit(n_splits=number_cross_validations, test_size=my_test_size, 
                          random_state = 711711)

  scores = cross_val_score(my_pipeline, data_X, data_Y, cv=cross_val,
                                scoring='f1_macro', n_jobs=1)
  print("%s: Using N=%d random splits and test_size=%0.2f, "
        "the average score was %0.2f with %0.4f std. dev." % 
        (name, number_cross_validations, my_test_size, scores.mean(), scores.std()))

# Score pipelines using default SVM with linear kernal
# Iterate through material and wear for both sensors
# and all hyperparams using the chosen window settings
# also test different test train splits
