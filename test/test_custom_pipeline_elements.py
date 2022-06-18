import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler

from custom_pipeline_elements import SampleScaler, ChannelScaler, FFTMag
import pdb

class ScalerTest(unittest.TestCase):

    def test_standard(self):
        x= [[2.2],[4.4],[0.5],[-10.0],[3.0],[8.0],[7.0],[9.0],[1.0],[23.0],[56.0],[0.0],[48.0]]
        scaler = StandardScaler()
        scaler.fit(x)
        z = scaler.transform(x)
        self.assertAlmostEqual(np.mean(z), 0.0, 4, "transformed mean not zero for standard")
        self.assertAlmostEqual(np.std(z), 1.0, 4, "transformed standard deveation for standard")

    def test_sample(self):
        x = np.random.rand(5,10) # 5 samples of 10 dimensions
        # print(x)
        scaler = SampleScaler() 
        scaler.fit(x)
        z = scaler.transform(x)
        #print (z)
        for y in z:
            self.assertAlmostEqual(np.mean(y), 0.0, 4, "transformed mean not zero for sample")
            self.assertAlmostEqual(np.std(y), 1.0, 4, "transformed standard deveation for sample")


    def test_channel(self):
       num_samples = 50
       num_dims    = 100
       x = np.random.rand(num_samples, num_dims)
       tf = ChannelScaler(4)
       tf.fit(x)
       z = tf.transform(x)
       self.assertEqual(z.shape[0], num_samples)
       self.assertEqual(z.shape[1], num_dims)
       for y in z:
         zz = y.reshape((4,-1), order='F')
         for yy in zz:
           self.assertAlmostEqual(np.mean(yy), 0.0, 4, "transformed mean not zero for chan.")
           self.assertAlmostEqual(np.std(yy), 1.0, 4, "transformed std. dev not one for chan.")

class FFTMagTest(unittest.TestCase):

  def test_FFT(self):
    num_samples = 50
    num_dims = 100
    x = np.random.rand(num_samples,num_dims)
    tf = FFTMag(4)
    tf.fit(x)
    z = tf.transform(x)
    self.assertEqual(z.shape[0], num_samples)
    self.assertGreaterEqual(z.shape[1], num_dims/2.)
    self.assertLessEqual(z.shape[1],num_dims)

if __name__ == '__main__':
    unittest.main()
