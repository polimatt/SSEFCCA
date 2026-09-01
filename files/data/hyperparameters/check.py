
from pickle import load
n = 'hyperparameters.pkl'
with open(n, 'rb') as f:
    hp = load(f)

print(hp)
