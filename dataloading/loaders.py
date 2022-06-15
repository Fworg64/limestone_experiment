import csv

from .data_file_paths import lcm_data_files, cap_base_path

# Data parameters
experiment_length_s = 4.0  # Length of rock cutting experiment from start in s
concrete_time_s     = 0.40 # Length of time spent cutting concrete at start of experiment
material_buffer_s   = 0.4  # length of time in seconds to throw away samples after concrete
initial_buffer_s    = 0.1  # length of time counted against concrete time to throw away first samples

wear_levels  = ["New", "Mod.", "Worn"]
penetrations = ["0.1 in.", "0.2 in.", "0.3 in."]
lines        = list(range(0,18))

lcm_dt = 1.0/537.6 # time between samples of force measurements made with LCM's strain gauges
cap_dt = 0.0025 # time in s between capacitive measurements

def load_lcm_file(input_file):
  """
  Opens a file recorded by the LCM, loads all data points into list of dictionaries
  """
  packet_list = []
  with open(input_file, "r") as infile:
    reader = csv.reader(infile, delimiter="	")
    for row in reader:
      packet = {"Sec": float(row[0]), "disp": float(row[1]), "v1": float(row[2]),
                "v2": float(row[3]), "v3": float(row[4]), "v4": float(row[5])}
      packet_list.append(packet)
  return packet_list

def load_strain_gauge_limestone(use_drag=False):
  """
  Returns list of time series data, lcm_X, and list of dictionary of labels, lcm_Y
  """
  lcm_X = []
  lcm_Y = []
  lcm_data = {}
  for wear in wear_levels:
    lcm_data[wear] = {}
    for pen in penetrations:
      # load lcm data here
      lcm_data[wear][pen] = {}
      for line in lcm_data_files[wear][pen].keys():
        # load data
        lcm_data[wear][pen][line] = load_lcm_file(lcm_data_files[wear][pen][line])
        # chop into samples and label
        if use_drag:
          sample_force = [conversions.calculate_drag_force(
                             point["v1"], point["v2"], point["v3"], point["v4"])/1000.0
                             for point in lcm_data[wear][pen][line]]
        else:
          sample_force = [[point["v1"], point["v2"], point["v3"], point["v4"]]
                              for point in lcm_data[wear][pen][line]]
          times = [point["Sec"] for point in lcm_data[wear][pen][line]]
          metadata = { "line_no": line, "pen": pen, "wear": wear}
          point_labels = label_split_material_for_data(times, metadata)
          lcm_X.extend(sample_force)
          lcm_Y.extend(point_labels)
  return lcm_X, lcm_Y

def label_split_material_for_data(times, labels):
  """
  Based on the time, determines the material over the course of the cutting pass.
  Labels each point of data with wear and material or nonclassified.
  Returns list of point labels with new material and wear labels
  data and times must be the same length, labels should be a dictionary of labels
  """
  point_labels = []
  for index, time in enumerate(times):
    matr = "NonClassified"
    wear = "NonClassified"
    if labels["line_no"] == 0 or labels["line_no"] == 17:
        pass # skip edge cases
    elif labels["wear"] == "New" and labels['pen'] == "0.3 in.": # restricted params for material
      if float(time) < initial_buffer_s:
        pass # skip edge cases
      elif float(time) < concrete_time_s:
        matr = "Concrete"
        wear = labels["wear"]
      elif float(time) < concrete_time_s + material_buffer_s:
        wear = labels["wear"]
      elif float(time) < experiment_length_s:
        matr = "Limestone"
        wear = labels["wear"]
    else: # skip material, only wear for other settings
      if float(time) < initial_buffer_s:
        pass # skip edge cases
      elif float(time) < experiment_length_s:
        wear = labels["wear"]
    point_labels.append(labels.copy())
    point_labels[-1]["mat."] = matr
    point_labels[-1]["wear"] = wear
  return point_labels
    

def load_datafile_to_lol(file_name, start_time):
  """
  Loads capacitive sensor data file to list of lists (lol) and return it
  """
  data = []
  times = []
  with open(cap_base_path + file_name) as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
      if float(row[0]) < float(start_time):
        continue
      elif float(row[0]) < float(start_time) + experiment_length_s:
        data.append(row[1:]) # drop time, keep chan's a-d 
        times.append(float(row[0]) - float(start_time)) # catch time
      else:
        break
  return data, times

def load_cap_limestone():
  """
  Returns list of time series data, cap_X, and list of dictionary of labels, cap_Y
  """
  cap_X = []
  cap_Y = []
  #data_files = []
  classifications_headers = ["file_name", "start_time", "wear", "pen", "line_no"]
  with open(cap_base_path + "classifications.csv") as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
      # load meta-data
      meta_dict = {classifications_headers[i]: row[i] 
                   for i in range(len(classifications_headers))}
      data, times = load_datafile_to_lol(meta_dict["file_name"], meta_dict["start_time"])
      point_labels = label_split_material_for_data(times, meta_dict)
      cap_X.extend(data)
      cap_Y.extend(point_labels)
  return cap_X, cap_Y


