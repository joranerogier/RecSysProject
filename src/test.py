from sdv import load_demo
from sdv import SDV

metadata, tables = load_demo(metadata=True)

"""
Fit a model: buid a hierarchical statistical model of the data.
During this process, SDV will traverse across all the tables in your dataset following 
the primary key-foreign key relationships and learn the probability distributions 
of the values in the columns.
"""
sdv = SDV()
sdv.fit(metadata, tables)

"""
Sample data from the fitted model to generate new synthetic data.
"""
sampled = sdv.sample_all(5)

"""
Saving the model.
The generated pkl file will not include any of the original data in it, so it can be safely
sent to where the synthetic data will be generated without any privacy concerns.
"""
sdv.save('sdv.pkl')

"""
Loading the model.
"""
sdv = SDV.load('sdv.pkl')

