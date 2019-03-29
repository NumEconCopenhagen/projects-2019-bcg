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

#%%
# Adding a file from OECD containing eaech country with its coresponding country code
# Here we should rename it country_codes='./Data/ccode.xls'
country_codes="D://MScEconomics//Python//Project1//ccode.xls"
ccode=pd.read_excel(country_codes)
ccode.rename(columns = {'Country':'country', 'CODE':'Country Code'}, inplace=True)
ccode.head

#%%
# Merging two DataFrames: country code and gdp growth
GDP_ccode = pd.merge(gdp, ccode, how='outer',on=['country'])
GDP_ccode.rename(columns = {'year':'Year'}, inplace = True)
GDP_ccode['Country Code'].unique()
GDP_ccode.head()


#%%
#Import Average wage data
avg_wage = "D://MScEconomics//Python//Project1//OECDwage.csv"
wages= pd.read_csv(avg_wage)
wages.head()

#Cleaning the data- Dropping unnecessary columns from the wages dataset
drop_these = ['INDICATOR', 'SUBJECT' ,'MEASURE', 'FREQUENCY', 'Flag Codes']
wages.drop(drop_these, axis=1, inplace=True)
wages.isna().sum() # this should return numbers of null values but returns 0. Why?

#Rename the remaining columns, round the wage values and calculate the Average wage in percentages(%)
wages.rename(columns = {'LOCATION':'Country Code', 'TIME':'Year', 'Value':'Average Wage'}, inplace = True)
wages["Average Wage"].apply(np.round)
wages["Average Wage Growth (%)"] = wages['Average Wage'].pct_change()*100
wages.head()


#%%
#Uploading the total unemployment data 
total_unemployment = "D:\\MScEconomics\\Python\\Project1\\unempl.csv"
tot_unempl= pd.read_csv(total_unemployment)

#check na
tot_unempl.isna().sum()

#dropping unnecesariliy columns
drop_these = ['INDICATOR', 'SUBJECT' , 'MEASURE', 'FREQUENCY', 'Flag Codes']
tot_unempl.drop(drop_these, axis=1, inplace=True)

#rename the remainig columns
tot_unempl.rename(columns = {'LOCATION':'Country Code', 'TIME':'Year', 'Value':'Total Unemployment (%)'}, inplace = True)
tot_unempl.head()


#%%
#Adding inflation rate data from OECD databaste
inflation_rate = "D:\\MScEconomics\\Python\\Project1\\inflation.csv"
inflation = pd.read_csv(inflation_rate)

#Dropping unncesarily columns
drop_columnsi = ['INDICATOR', 'SUBJECT' , 'MEASURE', 'FREQUENCY', 'Flag Codes']
inflation.drop(drop_columnsi, axis=1, inplace=True)
inflation.isna().sum()

# Rename columns
inflation.rename(columns = {'LOCATION':'Country Code', 'TIME':'Year', 'Value':'Inflation Rate (%)'}, inplace = True)
inflation.head()


#%%
# Merging two DataFrames: wage and unemployment
wage_unempl = pd.merge(wages, tot_unempl, how='outer',on=['Country Code','Year'])
wage_unempl.head(10)

#%%
# Merging the above created DataFRame with the inflation. The result will
# be a data frame with wage growth(%), unemployment(%) and inflation(%)
wage_unempl_infl = pd.merge(wage_unempl, inflation, how='outer', on=['Country Code','Year'])
wage_unempl_infl.head()


#%%
# Creating the final DataFrame by merging wage_unemply_infl with GDP Growth DataFrame 
final = pd.merge(wage_unempl_infl, GDP_ccode, on = ['Country Code', 'Year'], how = 'left')
final.head()


#%%
# Reindexing the final Dataset
final = final.reindex(columns=['Year','country','Country Code','gdp growth', 'Inflation Rate (%)','Total Unemployment (%)', 'Average Wage', 'Average Wage Growth (%)'])
final.head(20)


#%%
# Dropping G20, EU28 and EA19 as we are not interested in these groups
for val in ['20','19','28']:
    I = final['Country Code'].str.contains(val)
    final = final.loc[I == False]

# Dropping countries that are not in OECD and checking that we are only
# left with OECD countries
for val in ['Brazil', 'Indonesia', 'South Africa', 'Colombia', 'China', 'India', 'Saudi Arabia', 'Argentina', 'Costa Rica']: 
    I = final['country'].str.contains(val)
    final = final.loc[I == False]   
final['country'].unique() 


#%%
# Deleting years that are not in range [2003,2017]
final = final[final.Year > 2002]
final = final[final.Year < 2018]


#%%
# Adding missing values for Switzerland-unemployment from a list
add = {'2003':'4.119999886','2004':'4.320000172', '2005':'4.440000057', '2006':'4','2007':'3.65000009', '2008':'3.349999905','2009':'4.119999886' }
for key, value in add.items():
   I = (final['country'] == 'Switzerland') & (final['Year'] == int(key))
   final.loc[I, ['Total Unemployment (%)']] = value

final[final['country'] == 'Switzerland'].head(10)

# Adding missing  values for Lithuania unemployment from a list
addL = {'2003':'12.86999989','2004':'10.68000031'}
for key, value in addL.items():
   I = (final['country'] == 'Lithuania') & (final['Year'] == int(key))
   final.loc[I, ['Total Unemployment (%)']] = value

# Adding missing  values for Turkey unemployment from a list    
addT =  {'2003':'10.53999996','2004':'10.84000015', '2005':'10.64000034'}
for key, value in addT.items():
   I = (final['country'] == 'Turkey') & (final['Year'] == int(key))
   final.loc[I, ['Total Unemployment (%)']] = value


#%% Display all rows that contain a null value
final[final.isnull().any(axis=1)]


#%% Converting "Unemployment Rate (%)"" column to float to be able to round the 
#numbers to two decimal points
final['Total Unemployment (%)'] = final['Total Unemployment (%)'].astype(float)
final.info()


#%% Renaming "country" and "gdp growth" for consistency
final.rename(columns = {'country':'Country', 'gdp growth':'GDP Growth (%)'}, inplace = True)
final.head(10)


#%% Converting all numerical data to two decimal numbers to make it sexy
final['GDP Growth (%)']=final['GDP Growth (%)'].round(2)
final['Inflation Rate (%)']=final['Inflation Rate (%)'].round(2)
final['Total Unemployment (%)']=final['Total Unemployment (%)'].round(2)
final['Average Wage']=final['Average Wage'].round(2)
final['Average Wage Growth (%)']=final['Average Wage Growth (%)'].round(2)
final.head(10)


#%% Exporting the final dataset in order to be used in the next stage of our
#project, using data to create nice graphs

final.to_csv('D:/MScEconomics/Python/Project1/thedata.csv', index = False)

# Importing the final dataset in order to work on it

thedata=pd.read_csv('D:/MScEconomics/Python/Project1/thedata.csv')
thedata.head()

#sss










