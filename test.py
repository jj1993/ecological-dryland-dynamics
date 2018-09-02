# def setup_function(func):
#     # The setup_function() function tests ensure that neither the yes.txt nor the
#     # no.txt files exist.
#     files = os.listdir('.')
#     if 'no.txt' in files:
#         os.remove('no.txt')
#     if 'yes.txt' in files:
#         os.remove('yes.txt')
#
# def teardown_function(func):
#     # The f_teardown() function removes the yes.txt file, if it was created.
#     files = os.listdir('.')
#     if 'yes.txt' in files:
#         os.remove('yes.txt')

# TODO: Everything in the testing

def test_internal():
    exp = (2.0 / np.pi) * (-2.0 / (3.0 * np.pi))
    obs = sinc2d(np.pi / 2.0, 3.0 * np.pi / 2.0)
    assert obs == exp

def test_edge_x():
    exp = (-2.0 / (3.0 * np.pi))
    obs = sinc2d(0.0, 3.0 * np.pi / 2.0)
    assert obs == exp

def test_edge_y():
    exp = (2.0 / np.pi)
    obs = sinc2d(np.pi / 2.0, 0.0)
    assert obs == exp
