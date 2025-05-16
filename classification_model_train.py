from classification import Classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns

# load the dataset
data = load_iris()
X, y = data.data, data.target

# splitting the data into training and testing data
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Classification()

# train the model on training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# evaluate the accuracy
print(f'Accuracy: {100*model.accuracy(y_test,y_pred):.2f}%')

# visualize the output
sns.scatterplot(x=X_test[:,0],y=y_pred,alpha=0.6)
plt.title("Predicted Classes vs. Feature 1")
plt.xlabel("Feature 1 (X_test[:,0])")
plt.ylabel("Predicted Class")
plt.show()