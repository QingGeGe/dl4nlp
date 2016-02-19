import unittest
from math import exp
from dl4nlp.utilities import *


class TestUtilities(unittest.TestCase):
    def assertDistribution(self, distribution):
        self.assertTrue(all(distribution >= 0.0))
        self.assertTrue(all(distribution <= 1.0))
        self.assertEqual(1.0, np.sum(distribution))

    def assertNumpyEqual(self, expect, actual):
        self.assertEqual(expect.shape, actual.shape)
        if expect.shape == ():  # This is scalar!
            self.assertAlmostEqual(expect, actual)
        else:   # This is array
            for e, a in zip(expect, actual):
                self.assertNumpyEqual(e, a)

    def test_softmax(self):
        # softmax should receive numpy array and return normalized vector
        expect = np.array([exp(1) / (exp(1) + exp(2)), exp(2) / (exp(1) + exp(2))])
        actual = softmax(np.array([1, 2]))
        self.assertDistribution(actual)
        self.assertNumpyEqual(expect, actual)

        # softmax should be invariant to constant offsets in the input
        # softmax should be able to handle very large or small values
        actual = softmax(np.array([1001, 1002]))
        self.assertNumpyEqual(expect, actual)
        actual = softmax(np.array([-1002, -1001]))
        self.assertNumpyEqual(expect, actual)

        # softmax should receive matrix and return matrix of same size
        expect = np.array([[exp(1) / (exp(1) + exp(2)), exp(2) / (exp(1) + exp(2))],
                           [exp(1) / (exp(1) + exp(2)), exp(2) / (exp(1) + exp(2))]])
        actual = softmax(np.array([[1, 2], [3, 4]]))
        self.assertNumpyEqual(expect, actual)

    def test_sigmoid(self):
        x = np.array([[1, 2], [-1, -2]])
        f = sigmoid(x)
        g = sigmoid_grad(f)
        expected = np.array([[0.73105858,  0.88079708],
                    [0.26894142,  0.11920292]])
        self.assertNumpyEqual(expected, f)

        expected = np.array([[0.19661193,  0.10499359],
                    [0.19661193,  0.10499359]])
        self.assertNumpyEqual(expected, g)

    def test_gradient_check(self):
        def quad(x):
            return np.sum(x ** 2), x * 2

        self.assertTrue(gradient_check(quad, np.array(123.456)))      # scalar test
        self.assertTrue(gradient_check(quad, np.random.randn(3,)))    # 1-D test
        self.assertTrue(gradient_check(quad, np.random.randn(4,5)))   # 2-D test

        def sigmoid_check(x):
            return sigmoid(x), sigmoid_grad(sigmoid(x))

        x = np.array(0.0)
        result = gradient_check(sigmoid_check, x)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
