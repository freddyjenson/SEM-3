
import matplotlib.pyplot as mtp
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values
print(x)

#finding optimal number of clusters using the elbow method

from sklearn.cluster import KMeans
wcss_list = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)
mtp.plot(range(1,11),wcss_list)
mtp.title('The Elbow method graph')
mtp.xlabel('Number of clusters(k)')
mtp.ylabel('wcss_list')
mtp.show()

#training k-means
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_predict = kmeans.fit_predict(x)
print(y_predict)

#visualizing cluster
mtp.scatter(x[y_predict == 0,0],x[y_predict == 0, 1], s=100, c = 'blue', label = 'Cluster1')
mtp.scatter(x[y_predict == 1,0],x[y_predict == 1, 1], s=100, c = 'green', label = 'Cluster2')
mtp.scatter(x[y_predict == 2,0],x[y_predict == 2, 1], s=100, c = 'red', label = 'Cluster3')
mtp.scatter(x[y_predict == 3,0],x[y_predict == 3, 1], s=100, c = 'cyan', label = 'Cluster4')
mtp.scatter(x[y_predict == 4,0],x[y_predict == 4, 1], s=100, c = 'orange', label = 'Cluster5')
mtp.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='black')

mtp.title('Clusters of customer')
mtp.xlabel('Annual Income (k$)')
mtp.ylabel('Spending Score (1-100)')
mtp.legend()
mtp.show()

