import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv('House_Rent_Dataset.csv')
print(data.head())

#check the data contains null values or not
print(data.isnull().sum())

#descriptive statistics of the data
print(data.describe())

#basic stats like mean, median,highest and lowest values of Rent of the house
print(f"Mean Rent: {data['Rent'].mean()}")
print(f"Median Rent: {data['Rent'].median()}")
print(f"Highest Rent: {data['Rent'].max()}")
print(f"Lowest Rent: {data['Rent'].min()}")

#now look at the rent of the houses in different cities according to no of bedrooms,halls and kitchens(BHK)
figure=px.bar(data,x=data["City"],y=data["Rent"],
color=data["BHK"],title="Rent in Different cities According to the BHK")
figure.show()

#now look at the rent of the houses in different cities according to area type
figure=px.bar(data,x=data["City"],y=data["Rent"],
color=data["Area Type"],title="Rent in Different cities According to the Area Type")
figure.show()

#now look at the rent of the houses in different cities according to furnishing status
figure=px.bar(data,x=data["City"],y=data["Rent"],
color=data["Furnishing Status"],title="Rent in Different cities According to the Furnishing Status")
figure.show()

#now look at the rent of the houses in different cities according to size
figure=px.bar(data,x=data["City"],y=data["Rent"],
color=data["Size"],title="Rent in Different cities According to the Size")
figure.show()

#Now let’s have a look at the number of houses available for rent in different cities according to the dataset:
cities=data["City"].value_counts()
labels=cities.index
counts=cities.values
colors=['gold','lightgreen']

fig=go.Figure(data=[go.Pie(labels=labels,values=counts,hole=0.5)])
fig.update_layout(title="Number of Houses Available for Rent in Different Cities")
fig.update_traces(hoverinfo='label+percent',textinfo='value',textfont_size=30,
marker=dict(colors=colors,line=dict(color='black',width=3)))
fig.show()

#preference for Tenant
tenant_preference=data["Tenant Preferred"].value_counts()
labels=tenant_preference.index
counts=tenant_preference.values
colors=['gold','lightgreen']


fig=go.Figure(data=[go.Pie(labels=labels,values=counts,hole=0.5)])
fig.update_layout(
    title={
        'text': "Preference for Tenant in India",
        'x':0.5,
        'xanchor': 'center'
    }
)
fig.update_traces(hoverinfo='label+percent',textinfo='value',textfont_size=30,
marker=dict(colors=colors,line=dict(color='black',width=3)))
fig.show()


data["Area Type"] = data["Area Type"].map({"Super Area": 1, 
                                           "Carpet Area": 2, 
                                           "Built Area": 3})
data["City"] = data["City"].map({"Mumbai": 4000, "Chennai": 6000, 
                                 "Bangalore": 5600, "Hyderabad": 5000, 
                                 "Delhi": 1100, "Kolkata": 7000})
data["Furnishing Status"] = data["Furnishing Status"].map({"Unfurnished": 0, 
                                                           "Semi-Furnished": 1, 
                                                           "Furnished": 2})
data["Tenant Preferred"] = data["Tenant Preferred"].map({"Bachelors/Family": 2, 
                                                         "Bachelors": 1, 
                                                         "Family": 3})
print(data.head())


#splitting data
from sklearn.model_selection import train_test_split
x = np.array(data[["BHK", "Size", "Area Type", "City", 
                   "Furnishing Status", "Tenant Preferred", 
                   "Bathroom"]])
y = np.array(data[["Rent"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.10, 
                                                random_state=42)

from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, 
               input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(xtrain,ytrain,batch_size=1,epochs=21)

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(xtrain,ytrain,batch_size=1,epochs=21)

print('Enter the House details to predict Rent')
a = int(input("Enter the No of BHK (eg:1):"))
b = int(input("Enter the Size of the house:"))
c = int(input("Enter the Area Type(super area = 1,Carpet Area = 2,Built Area =3)"))
d = int(input("Enter the Pin Code of the City: "))
e = int(input("Enter the Furnishing Status of the House (Unfurnished = 0, Semi-Furnished = 1, Furnished = 2): "))
f = int(input("Enter the Tenant Type (Bachelors = 1, Bachelors/Family = 2, Only Family = 3): "))
g = int(input("Enter the Number of bathrooms: "))
features = np.array([[a,b,c,d,e,f,g]])
print("Predicted House Price = ",model.predict(features))