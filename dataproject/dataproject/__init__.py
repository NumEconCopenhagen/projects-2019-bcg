import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import wb
import ipywidgets as widgets

#GDP growth data from World Bank

gdp = wb.download(indicator='NY.GDP.MKTP.KD.ZG', country=['all'], start=1997, end=2018)
gdp = gdp.rename(columns = {'NY.GDP.MKTP.KD.ZG':'gdp growth'})
gdp = gdp.reset_index()
gdp['year']=gdp['year'].astype(int) #datetime %year%
print(gdp)