from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        income = float(request.form['income'])
        num_purchases = float(request.form['num_purchases'])
        calls = float(request.form['calls'])

        df = pd.read_csv("data.csv")

        X = df[['Age', 'Income', 'NumPurchases', 'Calls']]
        y = df['Churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, zero_division=1)

        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Classification Report:\n{classification_rep}")

        new_data = [[age, income, num_purchases, calls]]
        new_df = pd.DataFrame(new_data, columns=['Age', 'Income', 'NumPurchases', 'Calls'])

        prediction = model.predict(new_df)

        result = "Клиент склонен к оттоку." if prediction[0] == 1 else "Клиент НЕ склонен к оттоку."
        return render_template('result.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
