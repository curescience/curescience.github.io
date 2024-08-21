from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam, Ftrl

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def runModel(X_train, y_train, X_test, y_test, num_columns):
    output = []

    # List of optimizer classes to test
    optimizer_classes = {
        'SGD': SGD,
        'RMSprop': RMSprop,
        'Adagrad': Adagrad,
        'Adadelta': Adadelta,
        'Adam': Adam,
        'Adamax': Adamax,
        'Nadam': Nadam,
        'Ftrl': Ftrl,
    }

    best_optimizer_name = None
    best_train_accuracy = 0

    # Model Results Section
    output.append('<h1 style="text-align: center; font-family: Barlow, sans-serif;">Model Results</h1>')

    # Table for Accuracy, Precision, Recall
    output.append('<table border="1" style="width: 100%; text-align: center; font-family: Barlow, sans-serif;">')
    output.append('<tr><th>Optimizer</th><th>Training Accuracy</th></tr>')

    for opt_name, optimizer_class in optimizer_classes.items():
        optimizer = optimizer_class()

        model = Sequential()
        model.add(Dense(units=16, activation='relu', input_shape=(num_columns,)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])

        model.fit(X_train, y_train, epochs=100, verbose=0)
        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)

        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            best_optimizer_name = opt_name

        output.append(f'<tr><td>{opt_name}</td><td>{train_accuracy:.4f}</td></tr>')

    output.append('</table>')

    # Best Optimizer Section
    output.append(f'<h3 style="text-align: center; font-family: Barlow, sans-serif;">Best Optimizer: {best_optimizer_name} with Training Accuracy: {best_train_accuracy:.4f}</h3>')

    best_optimizer = optimizer_classes[best_optimizer_name]()
    model = Sequential()
    model.add(Dense(units=16, activation='relu', input_shape=(num_columns,)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=best_optimizer, metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=100, verbose=0)

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    output.append(f'<h3 style="text-align: center; font-family: Barlow, sans-serif;">Test Accuracy: {test_accuracy:.4f}</p>')

    y_pred = model.predict(X_test)
    y_pred_classes = np.round(y_pred).astype(int).flatten()

    output.append('<h2 style="text-align: center; font-family: Barlow, sans-serif;">Classification Report</h2>')
    output.append('<table border="1" style="width: 100%; text-align: center; font-family: Barlow, sans-serif;">')
    output.append('<tr><th>Label</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>')

    report = classification_report(y_test, y_pred_classes, output_dict=True)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            output.append(f'<tr><td>{label}</td><td>{metrics["precision"]:.4f}</td><td>{metrics["recall"]:.4f}</td><td>{metrics["f1-score"]:.4f}</td><td>{metrics["support"]}</td></tr>')

    output.append('</table>')

    # Detailed Classification Results
    output.append('<h3 style="text-align: center; font-family: Barlow, sans-serif;">Detailed Classification Results</h3>')
    output.append('<table border="1" style="width: 100%; text-align: center; font-family: Barlow, sans-serif;">')
    output.append('<tr><th>Index</th><th>True Label</th><th>Predicted Label</th><th>Result</th></tr>')

    classification_details = [
        (i + 1, true_label, predicted_label, 'Correctly classified' if true_label == predicted_label else 'Incorrectly classified')
        for i, (true_label, predicted_label) in enumerate(zip(y_test, y_pred_classes))
    ]
    for i, true_label, predicted_label, result in classification_details:
        output.append(f'<tr><td>{i}</td><td>{true_label}</td><td>{predicted_label}</td><td>{result}</td></tr>')

    output.append('</table>')

    return "".join(output)

@app.route('/')
def index(message=None):
    return render_template('index.html', message=message)

@app.route('/tensorflow', methods=['POST'])
def upload_file():
    if 'file1' not in request.files or 'file2' not in request.files:
        return render_template('index.html', message="Please upload both files.")

    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1.filename == '' or file2.filename == '':
        return render_template('index.html', message="Both files must be selected.")

    if allowed_file(file1.filename) and allowed_file(file2.filename):

        # Read CSV files into DataFrames
        train_df = pd.read_csv(StringIO(file1.read().decode('utf-8')))
        test_df = pd.read_csv(StringIO(file2.read().decode('utf-8')))

        # Check for non-numerical values in the data (excluding column headers)
        def has_non_numerical_values(df):
            # Exclude the headers and check data cells
            return df.applymap(lambda x: isinstance(x, str) or not np.issubdtype(type(x), np.number)).iloc[1:].any().any()

        if has_non_numerical_values(train_df) or has_non_numerical_values(test_df):
            return render_template('index.html', message="Your data contains non-numerical values. Please fix this and re-upload.")

        if list(train_df.columns) != list(test_df.columns):
            return render_template('index.html', message="Train and test set are not compatible. Check that all columns in the training and testing data match.")

        num_columns = len(train_df.columns) - 1

        train_target_column = train_df.columns[-1]
        test_target_column = test_df.columns[-1]

        X_train = train_df.drop(train_target_column, axis=1).to_numpy()
        y_train = train_df[train_target_column].to_numpy()

        X_test = test_df.drop(test_target_column, axis=1).to_numpy()
        y_test = test_df[test_target_column].to_numpy()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        final_output = runModel(X_train, y_train, X_test, y_test, num_columns)
        return render_template('output.html', output=final_output)

    return render_template('index.html', message="Invalid file extension. Please upload *.csv files")


if __name__ == '__main__':
    app.run(debug=True)