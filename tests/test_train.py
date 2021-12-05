import unittest
import pathlib
import sys
import os
from contextlib import contextmanager

PACKAGE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, os.path.abspath(PACKAGE.parent))

class TestTrainScript(unittest.TestCase):

    def setUp(self):
        self.file_name = os.path.join(PACKAGE, 'Cube-1mm.stl')
        self.assertTrue(os.path.exists(self.file_name))
        with self.assertNotRaises(ImportError):
            from scripts import train
            self.main = train.main
            self.args = [self.file_name, '--max_epochs', '10', '--disable_log', '--quiet', '--number_sampling_points', '100']
        
    @contextmanager
    def assertNotRaises(self, exc_type):
        try:
            yield None
        except exc_type:
            raise self.failureException('{} raised'.format(exc_type.__name__))

    def test_lib(self):
        with self.assertNotRaises(ImportError):
            from scripts import test
        
        with self.assertNotRaises(Exception):
            test.main()

    def test_main(self):
        with self.assertNotRaises(Exception):
            self.main(self.args)

    def test_model_1(self):
        with self.assertNotRaises(Exception):
            self.args += ['--model', '1']
            self.main(self.args)

    def test_model_2(self):
        with self.assertNotRaises(Exception):
            self.args += ['--model', '2']
            self.main(self.args)

    def test_model_3(self):
        with self.assertNotRaises(Exception):
            self.args += ['--model', '3']
            self.main(self.args)

    def test_model_4(self):
        with self.assertNotRaises(Exception):
            self.args += ['--model', '4']
            self.main(self.args)

    def test_model_5(self):
        with self.assertNotRaises(Exception):
            self.args += ['--model', '5']
            self.main(self.args)

    def test_model_6(self):
        with self.assertNotRaises(Exception):
            self.args += ['--model', '6']
            self.main(self.args)

    def test_model_7(self):
        with self.assertNotRaises(Exception):
            self.args += ['--model', '7']
            self.main(self.args)
    
    def test_model_8(self):
        with self.assertNotRaises(Exception):
            self.args += ['--model', '8']
            self.main(self.args)

    def test_model_9(self):
        with self.assertNotRaises(Exception):
            self.args += ['--model', '9']
            self.main(self.args)
            
    def test_model_10(self):
        with self.assertNotRaises(Exception):
            self.args += ['--model', '10']
            self.main(self.args)

    def test_model_11(self):
        with self.assertNotRaises(Exception):
            self.args += ['--model', '11']
            self.main(self.args)

    def test_model_12(self):
        with self.assertNotRaises(Exception):
            self.args += ['--model', '12']
            self.main(self.args)

    def test_model_13(self):
        with self.assertNotRaises(Exception):
            self.args += ['--model', '13']
            self.main(self.args)

    def test_model_14(self):
        with self.assertNotRaises(Exception):
            self.args += ['--model', '14']
            self.main(self.args)

    def test_model_15(self):
        with self.assertNotRaises(Exception):
            self.args += ['--model', '15']
            self.main(self.args)

    def test_model_16(self):
        with self.assertNotRaises(Exception):
            self.args += ['--model', '16']
            self.main(self.args)

if __name__ == '__main__':
    unittest.main()