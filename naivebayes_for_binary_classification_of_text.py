from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Define training data
train_data = ["This is a positive text",
              "I love the new phone",
              "Amazing performance by the team",
              "Negative text, I hate it",
              "I am so disappointed with the service"]

# Define labels for training data
train_labels = [1, 1, 1, 0, 0]  # 1 for positive, 0 for negative

# Create a count vectorizer to convert text to numerical features
vectorizer = CountVectorizer()

# Transform the training data to numerical features
train_features = vectorizer.fit_transform(train_data)

# Train a naive Bayes classifier on the training data
clf = MultinomialNB()
clf.fit(train_features, train_labels)

# Define test data
test_data = ["I am happy with the product",
             "Terrible experience with the company",
             "Great service and fast delivery"]

# Transform the test data to numerical features
test_features = vectorizer.transform(test_data)

# Predict the labels for the test data
test_labels = clf.predict(test_features)

# Print the predicted labels and the accuracy of the classifier
print("Predicted labels:", test_labels)
print("Accuracy:", accuracy_score([1, 0, 1], test_labels))
