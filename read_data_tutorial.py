import urllib
import urllib2
import requests
from bs4 import BeautifulSoup
import gzip
import pickle
import os


url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
print 'downloading with urllib'
urllib.urlretrieve(url, 'mist.pkl.gz')
dir  = os.getcwd()

with gzip.open(os.path.join(dir,'mist.pkl.gz'), 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)

