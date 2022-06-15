import unittest

from custom_pipeline_elements import Windowizer, window_maker

class TestWindowizer(unittest.TestCase):

  def test_windowize_no_overlap(self):
    my_data = list(range(100))
    my_labels = [int(d > 50) for d in my_data]
    window_size = 7
    my_window = window_maker("boxcar", window_size)
    windower = Windowizer(my_window, 0.0)
    data, labels = windower.windowize(my_data, my_labels)
    self.assertEqual(len(data), len(labels))
    self.assertLessEqual(len(data), len(my_data)/window_size)
    print("Made %d samples and matching labels out of %.2f possible" % 
           (len(data), len(my_data)/window_size))
    