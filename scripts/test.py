from modules import Classifier
import numpy as np

rffile = "C:\\temp\\rf_pickle_1478.pickle"
print(rffile)
rf_model2 = Classifier.load_from_pickle(rffile)
print(rf_model2)


