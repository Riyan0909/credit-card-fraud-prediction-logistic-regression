import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

credit_card_data = pd.read_csv("creditcard.csv")

features = credit_card_data.drop("Class", axis=1)
fraud_labels = credit_card_data["Class"]

scaler = StandardScaler()

features = scaler.fit_transform(features)

features_train, features_test, labels_train, labels_test = train_test_split(features,fraud_labels,test_size=0.2,random_state=42)

fraud_detection_model = LogisticRegression(max_iter=1000)

fraud_detection_model.fit(features_train,labels_train)

fraud_predictions = fraud_detection_model.predict(features_test)

accuracy = accuracy_score(labels_test,fraud_predictions)

print("Model Accuracy:")
print(f"{accuracy * 100:.2f}%")

sample_transaction = features_test[10].reshape(1, -1)

prediction = fraud_detection_model.predict(
    sample_transaction
)


if prediction[0] == 1:
    print("Transaction Result: Fraud")
else:
    print("Transaction Result: Normal")