import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor

data=pd.read_csv("Project_test.csv")
data1=pd.read_csv("Project_train.csv")
print(data.info())
#mean median etc..
print(data.describe())
#checking whether any null values present in the data
print(data.isnull())
#sum the all null values in the whole data
print(data.isnull().sum())

plt.figure(dpi=125)
sns.heatmap(np.round(data1.corr(numeric_only=True),2),annot=True)
plt.show()

x=data1[['MonsoonIntensity','TopographyDrainage','RiverManagement','Deforestation','Urbanization','ClimateChange','DamsQuality','Siltation','AgriculturalPractices','Encroachments','IneffectiveDisasterPreparedness','DrainageSystems','CoastalVulnerability','Landslides','Watersheds','DeterioratingInfrastructure','PopulationScore','WetlandLoss','InadequatePlanning','PoliticalFactors']]
y=data1['FloodProbability']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model =DecisionTreeRegressor(criterion="squared_error",random_state=100,max_depth=3,min_samples_leaf=10)
model.fit(X_train, Y_train)
# Evaluate the model on validation
y_pred= model.predict(X_test)
text_representation=tree.export_text(model)
print(text_representation)
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X_train.columns,rounded=True)
plt.show()

mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)
print(f"mean squared error",mse)
print(f"r^2 error",r2)