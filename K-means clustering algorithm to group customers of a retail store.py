import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
data = {
    'Customer_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Total_Spend': [500, 1500, 200, 800, 1200, 300, 700, 1000, 400, 900],
    'Purchase_Frequency': [5, 12, 2, 8, 10, 3, 6, 11, 4, 9]
}

df = pd.DataFrame(data)
df.drop('Customer_ID', axis=1, inplace=True)  
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
k = 3  
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)
print(df)
plt.scatter(df['Total_Spend'], df['Purchase_Frequency'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Total Spend')
plt.ylabel('Purchase Frequency')
plt.title('Customer Segmentation')
plt.colorbar(label='Cluster')
plt.show()