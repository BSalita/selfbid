import unittest


class TestTest(unittest.TestCase):

    def test_test(self):
        x = 7
        self.assertTrue(x > 5)


if __name__ == '__main__':
    unittest.main()
