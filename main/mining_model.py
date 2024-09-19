import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize, StandardScaler, OneHotEncoder
from sklearn.metrics import \
    confusion_matrix, \
    accuracy_score, precision_score, recall_score, \
    f1_score, average_precision_score
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import KFold, cross_val_score, ShuffleSplit

start = time.time()
# Import data.csv
df = pd.read_csv('../data/data.csv')

# Preprocessing for association rules
for column_name in df.columns:
    column = df[column_name]
    count = (column == 0).sum()
    print('Count of zeros in column ', column_name, ' is : ', count)

# New DataSet without values o zero in FirstDose column
df3 = df.loc[(df[['FirstDose']] != 0).all(axis=1)]
print("This is shape for the dataset df3 : ", df3.shape)
z = pd.DataFrame(df3['TargetGroup'] + "," + df3['Vaccine'])
# Recreate column with name = 'Total'
z['Total'] = z[0]
# Drop column with name = '0' so we can have one single column on DataFrame z
z = z.drop(labels=0, axis=1)
# Transform DataFrame to list with comma seperated values
z = list(z['Total'].apply(lambda x: x.split(",")))

# View data
print("\nA glance for our DataFrame : \n", df.head(10))

# Delete column = 'FirstDoseRefused' because almost everything is missing
del df['FirstDoseRefused']
# Delete column = 'Reporting Country' because it is already administered to region
del df['ReportingCountry']
# Delete column = 'UnknownDose' because only 4385 doses where classified as unknown
# the remaining values == 0 hurt our model without provided useful info
del df['UnknownDose']
# Delete column = 'NumberDosesExported' because correlation of doses received and
# exported does not exit in the data set
# del df['NumberDosesExported']
del df['YearWeekISO']


# See the different values in column = 'Vaccine'
print('\nPreview Of the frequency of Vaccine.values : \n',
      df['Vaccine'].value_counts().sort_values(ascending=False).head(17))
df['Vaccine'] = [0 if x == 'COM'
                 else 1 if x == 'MOD'
                 else 2 if x == 'AZ'
                 else 3 if x == 'JANSS'
                 else 4 for x in df['Vaccine']]

# View changes
print('\nNumeric transformation of  Vaccine Label : \n', df['Vaccine'].head(5))

# View data types
print('\nData Types of DataFrame columns : \n', df.dtypes)

# Calculate how many categories there are in object columns
for col_name in df.columns:
    if df[col_name].dtypes == 'object':
        unique_cat = len(df[col_name].unique())
        print("\nFeature '{col_name}' has {unique_cat} unique categories".format(
            col_name=col_name, unique_cat=unique_cat))

# Convert data_type =='object' column with to int values.
region_map = df['Region'].value_counts().to_dict()
df['Region'] = df['Region'].map(region_map)
print('\n\nPrint Region values after conversion : ', df['Region'])

# Create a small list containing column names whose dtypes == object
todummy_list = ['TargetGroup']

# Define function dummy  columns in order to deal with object data
def dummy_x(X, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(X[x], prefix=x, dummy_na=False)
        X = X.drop(x, 1)
        X = pd.concat([X, dummies], axis=1)
    return X

# Create dummy columns
df = dummy_x(df, todummy_list)

# View results
print("\nAfter dummy columns creation : \n\n", df.head(10))
print("\nShape of DataFrame now is : ", df.shape)

# View which of the data is actually missing
print('\nList of null values in Dataframe : \n', df.isnull().sum())
imp = SimpleImputer(strategy='constant')
imp.fit(df)
df = pd.DataFrame(data=imp.transform(df), columns=df.columns)

# Review missing values
print('\n Missing values for out DataFrame now are : {} \n'.format(df.isnull().sum().sum()))

# Drop Vaccine label to prepare our data
df1 = df.drop(labels='Vaccine', axis=1)


# Normalize data
data_scaled = normalize(df1)
data_scaled = pd.DataFrame(data_scaled, columns=df1.columns)

# Assign targets and features values
y = df.Vaccine
X = data_scaled

# Decomposition with PCA
pc = PCA().fit(X.values)
pca = PCA(n_components=9, whiten=True)
principalComponents = pca.fit_transform(X.values)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1',
                                    'principal component 2',
                                    'principal component 3',
                                    'principal component 4',
                                    'principal component 5',
                                    'principal component 6',
                                    'principal component 7',
                                    'principal component 8',
                                    'principal component 9'])
# Final DataFrame displayed with the dimensionality reduction added
finalDf = pd.concat([principalDf, y], axis=1)
print("\nFinal DataFrame after Dimensionality Reduction : \n\n", finalDf.head())

plt.plot(np.cumsum(pc.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title('Estimated Number of Components needed to explain DF')
plt.show()

# Shuffle DataFrame Rows
new_df = finalDf.sample(frac=1, random_state=42)
# Prepare the data
features = new_df.drop(['Vaccine'], axis=1)
labels = pd.DataFrame(new_df['Vaccine'])
# Hold the value of features and labels
feature_array = features.values
label_array = labels.values

# Splitting the feature array and label array keeping 80% for the training sets
X_train, X_test, y_train, y_test = train_test_split(
    feature_array, label_array, test_size=0.09,
    stratify=labels, random_state=2022)

X_train = normalize(X_train)
X_test = normalize(X_test)

# Use ravel because fitting in model requires it
y_train = y_train.ravel()
y_test = y_test.ravel()

kf = KFold(n_splits=5)
# Define model RandomForest Classifier for Cross-Validation
rf = RandomForestClassifier(n_estimators=10, max_depth=1000, n_jobs=-1)
# Define model RandomForest Classifier
RF = RandomForestClassifier(n_estimators=10,
                            max_depth=1000,
                            random_state=0, n_jobs=-1).fit(X_train, y_train)
# Predict y values for X_test input
y_pred_RF = RF.predict(X_test)

# Use confusion matrix to view performance of model
conmat_RF = confusion_matrix(y_test, y_pred_RF)
val = np.mat(conmat_RF)
classnames = list(set(y_train))

df_cm_RF = pd.DataFrame(
    val, index=classnames, columns=classnames,
)

print("\nConfusion matrix for RF model : \n", df_cm_RF)
# Provide a plot figure heatmap for the above matrix
plt.figure()
heatmap = sns.heatmap(df_cm_RF/np.sum(df_cm_RF), annot=True, cmap="Blues")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.yaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Churn Random Forest Model Results')
plt.show()

# Accuracy Precision recall and F1 score
print('\nAccuracy for Random Forest Classifier : ',
      round(accuracy_score(y_test, y_pred_RF), 4))

print('Precision for Random Forest Classifier Classifier : ',
      round(precision_score(y_test, y_pred_RF, average='micro'), 4))
print('Recall for Random Forest Classifier Classifier : ',
      round(recall_score(y_test, y_pred_RF, average='micro'), 4))
print('F1 Score for Random Forest Classifier Classifier : ',
      round(f1_score(y_test, y_pred_RF, average='micro'), 4))

'''
# Cross Validation
shuffle_split = ShuffleSplit(test_size=0.3, train_size=0.5, n_splits=10)
score = cross_val_score(rf, X_train, y_train, cv=shuffle_split)
print("\n\nCross Validation Scores are : \n{}".format(score))
print("\n\nAverage Cross Validation score :{}".format((score.mean())))
'''

# -----------------------------------------------K_Nearest_Neighbours_Classification ----------------------------------
# Knn
neighbours = np.arange(1, 5)
train_accuracy = np.empty(len(neighbours))
test_accuracy = np.empty(len(neighbours))
for i, k in enumerate(neighbours):
    # Set up a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree", n_jobs=-1)

    # Fit the model
    knn.fit(X_train, y_train)

    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    # Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test)

idx = np.where(test_accuracy == max(test_accuracy))
x = neighbours[idx]

# k_nearest_neighbours_classification
knn = KNeighborsClassifier(n_neighbors=x[0], algorithm="kd_tree", n_jobs=-1)
knn.fit(X_train, y_train)
knn_predicted_test_labels = knn.predict(X_test)

# Use confusion matrix to view performance of model
conmat_KNN = confusion_matrix(y_test, knn_predicted_test_labels)
vl = np.mat(conmat_KNN)
df_cm_KNN = pd.DataFrame(
    vl, index=classnames, columns=classnames,
)

print("\n\nConfusion matrix for RF model : \n", df_cm_KNN)
plt.figure()

heat_map = sns.heatmap(df_cm_KNN/np.sum(df_cm_KNN), annot=True, cmap="Blues")

heat_map.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heat_map.yaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Churn K Nearest Neighbors Model Results')
plt.show()

# Accuracy precision recall and f1_score for K-NN classifier
knn_accuracy_score = round(accuracy_score(y_test, knn_predicted_test_labels), 4)
print("\nAccuracy Of K-NN For The Given Dataset : ", knn_accuracy_score)
print('Precision for K-NN Classifier : ',
      round(precision_score(y_test, knn_predicted_test_labels, average='micro'), 4))
print('Recall for K-NN Classifier : ',
      round(recall_score(y_test, knn_predicted_test_labels, average='micro'), 4))
print('F1 Score for K-NN Classifier : ',
      round(f1_score(y_test, knn_predicted_test_labels, average='micro'), 4))

# ------------------------------------- Clustering -----------------------------------------------
# --------------------------------------K-Means ------------------------------------------------------
# Clustering

y = y.astype(int)
# Create model with no. of clusters = 5
model = KMeans(n_clusters=5, random_state=42)
# Fit a huge portion of our data set inside the algorithm
data_kmeans = model.fit(X)
# Display the predicted clusters
print("\nPredicted KMeans clusters for dataframe : ", data_kmeans.labels_)
# Create a confusion matrix in order to evaluate performance of kmeans
conf_matrix = confusion_matrix(y, data_kmeans.labels_)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center',
                ha='center', size='xx-large')
plt.title('K-Means Clustering Confusion Matrix', fontsize=18)
plt.xlabel('Prediction', fontsize=16)
plt.ylabel('Actual', fontsize=16)
plt.show()

# ------------------------------------------ DBSCAN------------------------------------------------------
# Creating an object of the NearestNeighbors class

neighb = NearestNeighbors(n_neighbors=2, n_jobs=-1)
# Fitting the data to the object
nbrs = neighb.fit(X_train[:20000])
# Finding the nearest neighbours
distances, indices = nbrs.kneighbors(X_train[:20000])

# Sort and plot the distances results
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.title('Estimated numbers of eps shown at maximum curvature')
plt.rcParams['figure.figsize'] = (8, 6)
plt.plot(distances)
plt.show()

y_test = y_test.astype(int)
# Create Density-Based clustering model and fit X_test
DBSCAN_cluster = DBSCAN(eps=0.4, min_samples=13,
                        n_jobs=-1,
                        ).fit(X_train[:20000])
# Estimated number of Clusters
no_clusters = len(np.unique(DBSCAN_cluster.labels_))
# Estimated number of noise points
no_noise = np.sum(np.array(DBSCAN_cluster.labels_) == -1, axis=0)
# Display results
print("\nEstimated no. of clusters for DBSCAN: %d" % no_clusters)
print("\nEstimated no. of noise points DBSCAN: %d" % no_noise)
print("\nClusters created are : ", DBSCAN_cluster.labels_)
# Create a confusion matrix in order to evaluate performance of kmeans
y_test = y_test.astype(int)
cm = confusion_matrix(y_train[:20000], DBSCAN_cluster.labels_)
ax1 = sns.heatmap(cm/np.sum(cm), annot=True,
                  fmt='.2%', cmap='Blues')

ax1.set_title('DBSCAN Confusion Matrix with labels\n\n');
ax1.set_xlabel('\nPredicted Vaccine Company')
ax1.set_ylabel('Actual Vaccine Company ');

# Display the visualization of the Confusion Matrix
plt.show()

# ------------------------------------- HierarchicalClustering ------------------------------------------
# Create model
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
# Fit X_test and predict clustering of its values
y_hc = hc.fit_predict(X_test)
# Display clustering results
print("\nPredicted Hierarchical Clustering clusters : ", y_hc)
# Evaluate the performance of the Hierarchical Clustering Mode
# by creating a confusion matrix
y_test = y_test.astype(int)
cf_matrix = confusion_matrix(y_test, y_hc)

ax2 = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')


ax2.set_title('Agglomerative Clustering  Confusion Matrix with labels\n\n');
ax2.set_xlabel('\nPredicted Values')
ax2.set_ylabel('Actual Values ');

ax2.xaxis.set_ticklabels(['AZ', 'COM', 'JANSS', 'MOD', 'Other'])
ax2.yaxis.set_ticklabels(['AZ', 'COM', 'JANSS', 'MOD', 'Other'])
# Display the visualization of the Confusion Matrix.
plt.show()

# --------------------------------------------------Association Rules----------------------------------------------
# One-hot encoding
te = TransactionEncoder()
data_transformed = te.fit(z).transform(z)
df2 = pd.DataFrame(data_transformed, columns=te.columns_)
df2 = df2.replace(False, 0)
df2 = df2.replace(True, 1)
# See transformed df2 dataframe
print(df2)

# Find frequent item sets
frequent_items = apriori(df2, min_support=0.02, use_colnames=False, verbose=0, low_memory=False)
print("\n", frequent_items, "\n")

# Let's view our interpretation values using the Association rules function.
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.02)
rules = rules.sort_values(by='confidence', ascending=False)
print("\n", rules.head(10))

# Support versus Confidence visualized
plt.figure()
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence')
plt.show()

# Support versus Lift visualized
plt.figure()
plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Lift')
plt.title('Support vs Lift')
plt.show()

# Lift versus Confidence visualized
fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'],
         fit_fn(rules['lift']))
plt.xlabel('Lift')
plt.ylabel('Confidence')
plt.title('Lift vs Confidence')
plt.show()
# End of cpu working time with the command entered below
end = time.time()
print("\n\nTime elapsed: %.3f seconds" % (end-start))  # CPU seconds elapsed