import pandas as pd

"""
    Divide the entire dataset into fake and real entries
"""

df = pd.read_csv("creditcard.csv")

real = df[df['Class'] == 0]
fake = df[df['Class'] == 1]

real.to_csv("real.csv")
fake.to_csv("fake.csv")
