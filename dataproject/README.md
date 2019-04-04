# Dataproject

Our porject's purpose is to provide easy access to the main economics indicators for all the 36 countries that are OECD members, for the period 2013-2017. More precisly we gather together data on the following indicators: GDP growth (%), Inflation Rate (%), Total Unemployment (%) and Average Wage Growth (%).

First, we importend GDP Growth data directly from the World Bank website, while data for the other indicators were downloaded as CSV files, locally, and then imported. After applying the neccesary cleaning and data structuring methods, and merging the different datasets together, the result was saved into the final dataset named "thedata". This dataframe is used for the next step, where we present the data.

The data presentation part includes three interactive graphs, as follows:

  1. An interactive plot where, using dropdown lists, the user can choose any of the two countries from our dataset and one of the economic indicators. In this way, the user will be able to compare countries, during a 15 years period, in terms of GDP Growth (%), Inflation Rate (%), Unemployment Rate (%) or Average Wage Growth (%).
  
  2. The second plot uses dropdown lists, with which you can select two indicators for the same country. This provides to the user very nice insights about the evolution and the relationship of the two selected indicators within a country and let us answer questions like: "What grew faster during the last 15 years, the Average Wage or the Inflation?" or "Does Unemployment Rate depend on GDP Growth?" etc.
  
  3. The third plot represents the famous Phillips curve and due to the plotting of the best fitted line, we can observe if there is any realtionship between Unemployment Rate and Inflation Rate. In our dataset, there is no significant relationship between the two variables.

