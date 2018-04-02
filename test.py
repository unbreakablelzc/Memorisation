import unittest
import filecmp
from lstm import DataGenerator
from HTMLTestRunner import HTMLTestRunner

class TestData(unittest.TestCase):

    def test_data_count(self):
        """Test method data_count(self, path) in class DataGenerator"""
        data_generator = DataGenerator()
        nb_zero, nb_one = data_generator.data_count("test data/data_balance_even.txt")
        self.assertEqual(nb_zero, 17)
        self.assertEqual(nb_one, 8)

    def test_data_balance(self):
        """Test method data_balance(self, path_solutions, path_train, path_test) in class DataGenerator"""
        data_generator = DataGenerator()
        data_generator.data_balance("test data/data_balance_even.txt",
                                    "test data/test_data_balance/traindata_even.txt",
                                    "test data/test_data_balance/testdata_even.txt")
        self.assertEqual(filecmp.cmp(r'test data/test_data_balance/traindata_even.txt',
                                     r'test data/even_balance_train.txt'), True)
        self.assertEqual(filecmp.cmp(r'test data/test_data_balance/testdata_even.txt',
                                     r'test data/even_balance_test.txt'), True)

        data_generator.data_balance("test data/data_balance_odd.txt",
                                    "test data/test_data_balance/traindata_odd.txt",
                                    "test data/test_data_balance/testdata_odd.txt")
        self.assertEqual(filecmp.cmp(r'test data/test_data_balance/traindata_odd.txt',
                                     r'test data/odd_balance_train.txt'), True)
        self.assertEqual(filecmp.cmp(r'test data/test_data_balance/testdata_odd.txt',
                                     r'test data/odd_balance_test.txt'), True)

    def test_data_extract(self):
        """Test method data_extract(self,  max_seq_len, path="") in class DataGenerator"""
        data_generator = DataGenerator()
        data_generator.data_extract(200, "test data/data_extract_real.txt")
        vector_data = [[[3542, 19, 3690], [3561, 43, 3716], [3604, 3, 3751], [3607, 19, 3754],[3626, 40, 3771],
                       [3666, 24, 3797],[3690, 9, 3845],[3699, 35, 3876],[3734, 10, 3915],[3744, 27, 3917],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]
        vector_labels = [[1.,0.]]
        vector_seqlen = [10]
        flag = (vector_data == data_generator.data).all()
        self.assertEqual(flag, True)
        self.assertEqual(vector_labels, data_generator.labels)
        self.assertEqual(vector_seqlen, data_generator.seqlen)

    def test_next_batch(self):
        """Test method next_batch(self, batch_size) in class DataGenerator"""
        data_generator = DataGenerator()
        data_generator.data_extract(200, "test data/data_next_batch_real.txt")
        data_generator.next_batch(500)
        self.assertEqual(100, data_generator.batch_id)
        self.id = 0
        data_generator.next_batch(50)
        self.assertEqual(50, data_generator.batch_id)

if __name__ == '__main__':
    unittest.main()
    """
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestData))

    with open('data/HTMLReport.html', 'w') as f:
        runner = HTMLTestRunner(stream=f,
                                title='MathFunc Test Report',
                                description='generated by HTMLTestRunner.',
                                verbosity=2
                                )
        runner.run(suite)
    """
