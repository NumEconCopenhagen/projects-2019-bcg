#%%
# A. Importing packages, necessary datasets and concluding to our final dataset 

# i. Importing the packages  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import wb
import ipywidgets as widgets

# ii. Dowloading data from the World Bank (Countries, Years and GDP Growth)
gdp = wb.download(indicator='NY.GDP.MKTP.KD.ZG', country=['all'], start=1997, end=2018)
gdp = gdp.rename(columns={'NY.GDP.MKTP.KD.ZG':'gdp growth'})
gdp = gdp.reset_index()
gdp['year'] = gdp['year'].astype(int) #datetime %year%
print(gdp)

# iii(a). The rest of the datasets were downloaded from OECD and imported manually
 
# iii(b). Data (Country code and Country)
## Here we should rename it country_codes='./Data/ccode.xls'
country_codes = "C:\\Users\\George\\Desktop\\Data Analysis Project\\Project1\\ccode.xls"
ccode=pd.read_excel(country_codes)
ccode.rename(columns = {'Country':'country', 'CODE':'Country Code'}, inplace=True)
ccode.head()

# iv. Merging the two imported DataFrames (ccode and gdp)
GDP_ccode = pd.merge(gdp, ccode, how='outer', on=['country'])
GDP_ccode.rename(columns={'year':'Year'}, inplace=True)
GDP_ccode['Country Code'].unique()
GDP_ccode.head()


#%%
# v(a). Data (Country Code, Year, Average wage and Average Wage Growth (%)) 
avg_wage = "C:\\Users\\George\\Desktop\\Data Analysis Project\\Project1\\OECDwage.csv"
wages = pd.read_csv(avg_wage)
wages.head()

# v(b). Cleaning the dataset, calculate the Average Wage Growth (%) and store it in a new column
drop_these = ['INDICATOR', 'SUBJECT' ,'MEASURE', 'FREQUENCY', 'Flag Codes']
wages.drop(drop_these, axis=1, inplace=True)
#wages.isna().sum() # maybe delete that line
wages.rename(columns = {'LOCATION':'Country Code', 'TIME':'Year', 'Value':'Average Wage'}, inplace=True)
wages["Average Wage"].apply(np.round)
wages["Average Wage Growth (%)"] = wages['Average Wage'].pct_change()*100
wages.head()

# vi(a). Data (Country Code, Year and Total Unemployment (%)) 
total_unemployment = "C:\\Users\\George\\Desktop\\Data Analysis Project\\Project1\\unempl.csv"
tot_unempl = pd.read_csv(total_unemployment)

#tot_unempl.isna().sum() #maybe delete that line

#vi(b). Cleaning and manipulating the data
drop_these = ['INDICATOR', 'SUBJECT' , 'MEASURE', 'FREQUENCY', 'Flag Codes']
tot_unempl.drop(drop_these, axis=1, inplace=True)
tot_unempl.rename(columns = {'LOCATION':'Country Code', 'TIME':'Year', 'Value':'Total Unemployment (%)'}, inplace=True)
tot_unempl.head()

# vii(a). Data (country Code, Year and Inflation Rate (%))
inflation_rate = "C:\\Users\\George\\Desktop\\Data Analysis Project\\Project1\\inflation.csv"
inflation = pd.read_csv(inflation_rate)

# vii(b). Cleaning and manipulating the data
drop_columnsi = ['INDICATOR', 'SUBJECT' , 'MEASURE', 'FREQUENCY', 'Flag Codes']
inflation.drop(drop_columnsi, axis=1, inplace=True)
#inflation.isna().sum() # maybe delete that line
inflation.rename(columns = {'LOCATION':'Country Code', 'TIME':'Year', 'Value':'Inflation Rate (%)'}, inplace=True)
inflation.head()

# viii. Merging the last three imported datasets (wages, tot_unempl and inflation)
wage_unempl = pd.merge(wages, tot_unempl, how='outer', on=['Country Code', 'Year'])
wage_unempl.head(10)

wage_unempl_infl = pd.merge(wage_unempl, inflation, how='outer', on=['Country Code','Year'])
wage_unempl_infl.head()

#%%
# ix(a). Creating the final dataset (after merging the two merged datasets) 
final = pd.merge(wage_unempl_infl, GDP_ccode, on=['Country Code', 'Year'], how='left')
final.head()

final = final.reindex(columns=['Year','country','Country Code','gdp growth', 'Inflation Rate (%)','Total Unemployment (%)', 'Average Wage', 'Average Wage Growth (%)'])
final.head(20)

# ix(b.) Dropping G20, EU28, EA19, non-OECD countries and years that are not in the range (2003, 2017)
for val in ['20','19','28']:
    I = final['Country Code'].str.contains(val)
    final = final.loc[I == False]

A = ['Brazil', 'Indonesia', 'South Africa', 'Colombia', 'China', 'India', 'Saudi Arabia', 'Argentina', 'Costa Rica']
for val in A: 
    I = final['country'].str.contains(val)
    final = final.loc[I == False]   
final['country'].unique() 

final = final[final.Year > 2002]
final = final[final.Year < 2018]

# ix(c). Adding missing values for unemployment (for Switzerland, Lithuania and Turkey)
# (data gained from World Bank)
add = {'2003':'4.119999886','2004':'4.320000172', '2005':'4.440000057', '2006':'4','2007':'3.65000009', '2008':'3.349999905','2009':'4.119999886' }
for key, value in add.items():
   I = (final['country'] == 'Switzerland') & (final['Year'] == int(key))
   final.loc[I, ['Total Unemployment (%)']] = value
final[final['country'] == 'Switzerland'].head(10)

addL = {'2003':'12.86999989','2004':'10.68000031'}
for key, value in addL.items():
   I = (final['country'] == 'Lithuania') & (final['Year'] == int(key))
   final.loc[I, ['Total Unemployment (%)']] = value

addT =  {'2003':'10.53999996','2004':'10.84000015', '2005':'10.64000034'}
for key, value in addT.items():
   I = (final['country'] == 'Turkey') & (final['Year'] == int(key))
   final.loc[I, ['Total Unemployment (%)']] = value

final[final.isnull().any(axis=1)] #Checking if there are any NaN values in our dataset
                                  #(NaN for Turkey - Average Wage and Average Wage Growth (%))

# ix(d). Converting "Unemployment Rate (%)" column to float type and rounding the columns with two decimals 
#numbers to two decimal points
final['Total Unemployment (%)'] = final['Total Unemployment (%)'].astype(float)
final.info()

final.rename(columns = {'country':'Country', 'gdp growth':'GDP Growth (%)'}, inplace=True) # Renaming "country" and "gdp growth" for consistency

final['GDP Growth (%)']=final['GDP Growth (%)'].round(2)
final['Inflation Rate (%)']=final['Inflation Rate (%)'].round(2)
final['Total Unemployment (%)']=final['Total Unemployment (%)'].round(2)
final['Average Wage']=final['Average Wage'].round(2)
final['Average Wage Growth (%)']=final['Average Wage Growth (%)'].round(2)
final.head(10)

# ix(e). Saving the final dataset for easier and faster use
final.to_csv('C:\\Users\\George\\Desktop\\Data Analysis Project\\Project1\\thedata.csv', index=False)


#%%
# B. Importing the created dataset (thedata) and ploting 

# i. Importing the dataset and renaming the columns
thedata=pd.read_csv('C:\\Users\\George\\Desktop\\Data Analysis Project\\Project1\\thedata.csv')
thedata.rename(columns = {'country':'Country', 'gdp growth':'GDP Growth (%)', '(%) AVG Wage':'AVG Wage Growth (%)'}, inplace=True)
thedata.head()

#%%
# ii. Plotting the comparison of a column between two countries
def _plot_1(thedata,Country1,Country2,Variable1):

    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(1,1,1)
    thedata.loc[:,['Year']] = pd.to_numeric(thedata['Year'])
                                                                
    I = (thedata['country'] == Country1)
    i = (thedata['country'] == Country2)
    
    x = thedata.loc[I,'Year']
    y = thedata.loc[I,Variable1]
    z = thedata.loc[i,Variable2]
    ax.plot(x,y,'g')
    ax.plot(x,z,'y')
    
    ax.set_xticks(list(range(2004, 2016 + 1, 2)))
    ax.set_xlabel('Year')
    ax.legend(loc='upper right')

def plot_1(thedata):
    
    widgets.interact(_plot_1,  
    thedata = widgets.fixed(thedata),
        Country1=widgets.Dropdown(
        description='OECD Country (No data for % AVG Wage for Turkey)', 
        options=thedata['country'].unique().tolist(),
        value='Australia',
        disabled=False),

        Country2=widgets.Dropdown(
        description='OECD Country (No data for % AVG Wage for Turkey)', 
        options=thedata['country'].unique().tolist(),
        value='Australia',
        disabled=False),

        Variable1 = widgets.Dropdown(
        description='Variable1', 
        options=['Total Unemployment (%)','Inflation Rate (%)','Average Wage Growth (%)','GDP Growth (%)'], 
        value='Total Unemployment (%)')

    )    
    
plot_1(thedata)

#%%
# iii. Plotting the trends of two columns in one country

def _plot_2(thedata,Country,variable1,variable2):

    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(1,1,1)
    thedata.loc[:,['Year']] = pd.to_numeric(thedata['Year'])
                                                                
    I = (thedata['Country'] == Country)
    
    x = thedata.loc[I,'Year']
    y = thedata.loc[I,variable1]
    z = thedata.loc[I,variable2]
    ax.plot(x,y,'g')
    ax.plot(x,z,'y')
    
    ax.set_xticks(list(range(2004, 2016 + 1, 2)))
    ax.set_xlabel('Year')
    ax.legend(loc='upper right')

def plot_2(thedata):
    
    widgets.interact(_plot_2,  
    thedata = widgets.fixed(thedata),
        Country=widgets.Dropdown(
        description='OECD Country (No data for % AVG Wage for Turkey)', 
        options=thedata['Country'].unique().tolist(),
        value='Australia',
        disabled=False),
                     
        variable1 = widgets.Dropdown(
        description='Variable1', 
        options=['Total Unemployment (%)','Inflation Rate (%)','Average Wage Growth (%)','GDP Growth (%)'], 
        value='GDP Growth (%)'),
                     
        variable2 = widgets.Dropdown(
        description='Variable2', 
        options=['Total Unemployment (%)','Inflation Rate (%)','Average Wage Growth (%)','GDP Growth (%)'], 
        value='Inflation Rate (%)')
        
    )                 

plot_2(thedata)

#%%
# iv. Plotting the Phillips Curve after choosing the country
def _philips_curve(thedata,Country):
 
    thedata.loc[:,['Year']] = pd.to_numeric(thedata['Year'])
    
    I = (thedata['Country'] == Country)
    
    a=thedata.loc[I,'Total Unemployment (%)']
    b=thedata.loc[I,'Inflation Rate (%)']
    plt.scatter(a,b)
    plt.xlabel('Total Unemployment (%)')
    plt.ylabel('Inflation Rate (%)')
    plt.title('Philips Curve')
    
    plt.plot(a, b, '--')

    YEAR=thedata['Year']
    
    plt.plot(np.unique(a), np.poly1d(np.polyfit(a, b, 1))(np.unique(a)))
    
    #for i, txt in enumerate(YEAR):
     #   plt.annotate(txt,(a[i], b[i]))
    
def philips_curve(thedata):
    
    widgets.interact(_philips_curve,  
    thedata = widgets.fixed(thedata),
        Country=widgets.Dropdown(
        description='OECD Country', 
        options=thedata['Country'].unique().tolist(),
        value='Australia',
        disabled=False)
                    )
                    
philips_curve(thedata)


