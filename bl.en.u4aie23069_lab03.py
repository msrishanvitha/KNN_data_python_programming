import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 🔹 A1: Data Preparation and Class Splitting
data = np.array([
    [20, 6, 2, 386],
    [16, 3, 6, 289],
    [27, 6, 2, 393],
    [19, 1, 2, 110],
    [24, 4, 2, 280],
    [22, 1, 5, 167],
    [15, 4, 2, 271],
    [18, 4, 2, 274],
    [21, 1, 4, 148],
    [16, 2, 4, 198]
])

class1 = data[:5, :-1]
class2 = data[5:, :-1]

centroid1 = class1.mean(axis=0)
centroid2 = class2.mean(axis=0)

spread1 = class1.std(axis=0)
spread2 = class2.std(axis=0)

distance = np.linalg.norm(centroid1 - centroid2)

print("Centroid 1:", centroid1)
print("Spread 1:", spread1)
print("Centroid 2:", centroid2)
print("Spread 2:", spread2)
print("Interclass Distance:", distance)

# 🔹 A2: Histogram of a Feature (Candies)
feature = data[:, 0]

plt.hist(feature, bins=5, edgecolor='black')
plt.xlabel("Candies (#)")
plt.ylabel("Frequency")
plt.title("Histogram of Candies Purchased")
plt.show()

mean = np.mean(feature)
variance = np.var(feature)

print("Feature Mean:", mean)
print("Feature Variance:", variance)

# 🔹 A3: Minkowski Distance Plot
vec1 = data[0, :-1].astype(float)
vec2 = data[1, :-1].astype(float)

r_values = range(1, 11)
distances = [minkowski(vec1, vec2, r) for r in r_values]

plt.plot(r_values, distances, marker='o')
plt.xlabel("Minkowski Parameter r")
plt.ylabel("Distance")
plt.title("Minkowski Distance for Different r")
plt.show()

# 🔹 A4: Train-Test Split (Categorizing Payment)
X = data[:, :-1]
y = np.where(data[:, -1] > 250, 1, 0)  # 1 = High Payment, 0 = Low Payment

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🔹 A5: Train kNN Classifier (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 🔹 A6: Accuracy of kNN
accuracy = knn.score(X_test, y_test)
print("kNN Accuracy:", accuracy)

# 🔹 A7: Predictions
predictions = knn.predict(X_test)
print("Predictions:", predictions)

# 🔹 A8: Accuracy for k from 1 to 11
k_values = range(1, 12)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracies.append(knn.score(X_test, y_test))

plt.plot(k_values, accuracies, marker='o')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("kNN Accuracy vs k")
plt.show()

# 🔹 A9: Confusion Matrix and Performance Metrics
y_pred = knn.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
