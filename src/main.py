# TODO: add arguments to change variables directly.
import pandas as pd

# Import own packages
from fit_CTGAN import TrainModel
from load_input_data import InputDataLoader

data_loader = InputDataLoader("ml-100k")
data = data_loader.get_data()

m = TrainModel(dn="test")
m.build_model(data)
#m.eval_model(data)