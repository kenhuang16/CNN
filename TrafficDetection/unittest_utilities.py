"""
unitest
"""
import unittest

from utilities import non_maxima_suppression

class TestUtilities(unittest.TestCase):
    """Test functions in utilities.py"""
    def test_non_maxima_suppression(self):
        """test function non_maxima_suppression()"""
        boxes = (((5, 5), (15, 15)),
                 ((6, 6), (16, 16)),
                 ((7, 7), (17, 17)),
                 ((18, 18), (28, 28)))
        scores = (0.5, 0.6, 0.7, 0.2)

        output = non_maxima_suppression(boxes, scores, threshold=0.5)
        self.assertEqual(output, [((7, 7), (17, 17)), ((18, 18), (28, 28))])


if __name__ == '__main__':
    unittest.main(verbosity=2)