# Assuming you have a dataset with features and labels
# Replace `X` with your features and `y` with your labels

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Import the model and any necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

# Create a pipeline consisting of a TF-IDF vectorizer and a Random Forest classifier
model = make_pipeline(
    TfidfVectorizer(),  # Convert text data into numerical features using TF-IDF
    RandomForestClassifier(n_estimators=100, random_state=42)  # Train a Random Forest classifier
)

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
