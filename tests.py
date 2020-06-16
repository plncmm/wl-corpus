import unittest
import src

class TestConfidenceInterval(unittest.TestCase):
    def setUp(self):
        self.test_data = [ 2.30672566,  3.31442004,  0.47613732,  0.54143721,  1.01833591, 0.16258758, -0.35041846, -0.38756352,  0.84095111,  2.01866616]
    def test_mean_confidence_interval_mean(self):
        self.assertAlmostEqual(src.mean_confidence_interval(self.test_data)[0], 0.994, places=2)
    def test_mean_confidence_interval_lower(self):
        self.assertAlmostEqual(src.mean_confidence_interval(self.test_data)[1], 0.132, places=2)
    def test_mean_confidence_interval_upper(self):
        self.assertAlmostEqual(src.mean_confidence_interval(self.test_data)[2], 1.855, places=2)

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.test_string="paciente presenta hipertensión arterial, dm y mal de chagas"
    def test_tokenizer(self):
        self.assertEqual(src.tokenizer(self.test_string),["paciente","presenta","hipertensión","arterial",",","dm","y","mal","de","chagas"])
if __name__ == '__main__':
    unittest.main()