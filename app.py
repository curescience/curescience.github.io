from flask import Flask, request, render_template, redirect, url_for, send_file
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam, Ftrl
import pandas as pd
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from cns_mpo_csv_to_df import CNS_MPO_csv_to_df

app = Flask(__name__)

# Store the optimizer and training data in global variables
training_data = None
optimizer_classes = {
    'SGD': SGD,
    'RMSprop': RMSprop,
    'Adagrad': Adagrad,
    'Adadelta': Adadelta,
    'Adam': Adam,
    'Adamax': Adamax,
    'Nadam': Nadam,
    'Ftrl': Ftrl
}

# Function to validate CSV data
def validate_csv_data(df):
    """ Validate that all values in the CSV are numeric except for the header row """
    try:
        # Convert all columns to numeric, forcing errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        # Check if any NaNs are present (indicating non-numeric data)
        return df.notna().all().all()
    except Exception as e:
        print(f"Error validating CSV data: {e}")
        return False

# Home route - redirect to the training route
@app.route('/')
def home():
    return redirect(url_for('index'))

# Training section
@app.route('/tensorflow', methods=['GET', 'POST'])
def index():
    global training_data
    if request.method == 'POST':
        file = request.files['train_file']
        if file:
            try:
                # Load training data
                train_data = pd.read_csv(file)
                
                # Validate CSV data
                if not validate_csv_data(train_data):
                    return render_template('index.html', error="Invalid data: Training dataset must contain only numerical values.")

                X_train = train_data.iloc[:, :-1].values
                y_train = train_data.iloc[:, -1].values
                training_data = (X_train, y_train)
                
                # Prepare result info for all optimizers
                results = []
                for name, optimizer_class in optimizer_classes.items():
                    optimizer = optimizer_class()  # Create a new instance of the optimizer
                    
                    model = Sequential()
                    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
                    model.add(Dense(1, activation='sigmoid'))
                    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
                    
                    # Train the model and evaluate training accuracy
                    model.fit(X_train, y_train, epochs=10, verbose=0)
                    _, training_accuracy = model.evaluate(X_train, y_train, verbose=0)
                    
                    results.append({
                        'optimizer': name,
                        'accuracy': f"{training_accuracy*100:.2f}%",
                        'model_path': url_for('download_model', optimizer_name=name)
                    })
                
                # Pass the results to output.html for rendering
                return render_template('output.html', results=results)

            except Exception as e:
                return render_template('index.html', error=f"Error processing file: {e}")

    return render_template('index.html')

# Route to generate and download the model on-demand
@app.route('/download_model/<optimizer_name>')
def download_model(optimizer_name):
    global training_data
    if training_data is None:
        return redirect(url_for('index'))

    X_train, y_train = training_data
    
    optimizer_class = optimizer_classes.get(optimizer_name)
    if optimizer_class is None:
        return redirect(url_for('index'))
    
    optimizer = optimizer_class()  # Create a new instance of the optimizer
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10, verbose=0)
    
    model_path = f'model_{optimizer_name.lower()}.keras'
    model.save(model_path)
    
    response = send_file(model_path, as_attachment=True)
    
    # Remove model file after sending
    os.remove(model_path)
    
    return response

# Evaluation section
@app.route('/evaluate', methods=['POST'])
def evaluate():
    model_file = request.files['model_file']
    test_file = request.files['test_file']
    
    test_error = None
    if model_file and test_file:
        try:
            # Load model
            model_path = 'temp_model.keras'
            model_file.save(model_path)
            model = load_model(model_path)
            
            # Load test data
            test_data = pd.read_csv(test_file)
            
            # Validate CSV data
            if not validate_csv_data(test_data):
                return render_template('index.html', error="Invalid data: Testing dataset must contain only numerical values.")

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            # Validate CSV attributes
            if X_test.shape[1] != model.input_shape[1]:
                return render_template('index.html', test_error="Invalid data: Testing dataset and inputted model are incompatible.")

            # Evaluate model
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            y_pred = (model.predict(X_test) > 0.5).astype(int)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            confusion = confusion_matrix(y_test, y_pred)
            
            # Prepare data for rendering
            classification_report_data = []
            for label, metrics in report.items():
                if label != 'accuracy' and label != 'weighted avg':  # Exclude 'weighted avg' from the main report
                    classification_report_data.append({
                        'label': label,
                        'precision': f"{metrics['precision']:.2f}",
                        'recall': f"{metrics['recall']:.2f}",
                        'f1_score': f"{metrics['f1-score']:.2f}",
                        'support': metrics['support']
                    })
            weighted_avg = report.get('weighted avg', {})
            weighted_avg = {
                'precision': f"{weighted_avg.get('precision', 0):.2f}",
                'recall': f"{weighted_avg.get('recall', 0):.2f}",
                'f1_score': f"{weighted_avg.get('f1-score', 0):.2f}",
                'support': weighted_avg.get('support', 0)
            }
            confusion_matrix_data = []
            for index, (true_label, pred_label) in enumerate(zip(y_test, y_pred.flatten())):
                result = 'Correctly Classified' if true_label == pred_label else 'Incorrectly Classified'
                confusion_matrix_data.append({
                    'index': index,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'result': result
                })
            
            # Remove temporary model file
            os.remove(model_path)
            
            return render_template('evaluation_results.html', accuracy=f"{accuracy*100:.2f}%", 
                                   classification_report=classification_report_data,
                                   weighted_avg=weighted_avg,
                                   confusion_matrix=confusion_matrix_data)
        except Exception as e:
            print(f"Error processing files: {e}")  # Debugging line
            return render_template('index.html', test_error=f"Error processing files: {e}")

    return redirect(url_for('index'))

@app.route('/generate', methods=['POST'])
def generate():
    if 'smiles_file' not in request.files:
        return render_template('index.html', error="File format is incorrect")
    
    file = request.files['smiles_file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")

    if file:
        try:
            # Process the file using CNS_MPO_csv_to_df
            cns_mpo_processor = CNS_MPO_csv_to_df(file)  # Ensure the file-like object is passed correctly
            resulting_df = cns_mpo_processor.to_dataframe()

            # Save the processed DataFrame to a CSV file
            output_file_path = 'descriptors.csv'
            resulting_df.to_csv(output_file_path, index=False)

            # Send the file back to the user for download
            return send_file(output_file_path, as_attachment=True)
        except Exception as e:
            return render_template('index.html', error=f"Error processing file: {e}")

    return redirect(url_for('index'))

# Handle favicon requests to prevent 404 error
@app.route('/favicon.ico')
def favicon():
    return send_file('static/favicon.ico')

if __name__ == '__main__':
    app.run(debug=True)
