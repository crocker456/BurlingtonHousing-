#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas 
import quandl
def parse_btv_props():
    #Read in csv from Burlington Open data
    df = pandas.read_csv("https://opendata.arcgis.com/datasets/276ccff527454496ac940b60a2641dda_0.csv")
    #Split the 'PropertyCenterPoint' column into 'Latitude' and 'Longitude' in a seperate dataframe
    df2 = df['PropertyCenterPoint'].str.strip('()').str.split(', ', expand=True).rename(columns={0:'Latitude', 1:'Longitude'})
    df3 = df['Baths'].str.split('/', expand=True).rename(columns={0:'Full_Baths', 1:'Half_Baths'})
    # create a dataframe with the new variables
    df = pandas.concat([df, df2, df3],axis=1)
    df.drop('PropertyCenterPoint', 1, inplace = True)
    df['Half_Baths'] = pandas.to_numeric(df['Half_Baths'])
    df['Full_Baths'] = pandas.to_numeric(df['Full_Baths'])
    df['SaleDate'] = pandas.to_datetime(df['SaleDate'])
    df['Year'] = df['SaleDate'].apply(lambda row: row.year)
    df = df[df['Year'] > 1996]
    df['Year'] = df['Year'].astype(int)
    mydata = quandl.get("FRED/VTNGSP", start_date="1931-01-01")
    mydata2 = quandl.get("FRED/VTSTHPI", start_date="1931-01-01")
    mydata = mydata.reset_index()
    mydata2 = mydata2.resample('A').mean()
    mydata2 = mydata2.reset_index()
    mydata['Year'] = mydata['Date'].apply(lambda row: row.year)
    mydata2['Year'] = mydata2['Date'].apply(lambda row: row.year)
    mydata = mydata.rename(columns = {'Value': 'VermontGDP'})
    mydata2 = mydata2.rename(columns = {'Value': 'VermontHPI'})
    mydata = mydata.drop('Date', 1)
    mydata2 = mydata2.drop('Date', 1)
    df = df.merge(mydata, 'left', on = 'Year')
    df = df.merge(mydata2, 'left', on = 'Year')
    print("Dataframe dimensions:", df.shape)
    return(df)
if __name__ == "__main__":
    parse_btv_props()