import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        print(request.form.get('Nitrogen'))
        try:
            N=float(request.form['N'])
            P=float(request.form['P'])
            K=float(request.form['K'])
            temperature=float(request.form['temperature'])
            humidity=float(request.form['humidity'])
            ph=float(request.form['ph'])
            rainfall=float(request.form['rainfall'])
            pred_args= [N,P,K,ph,rainfall]
            pred_args_arr = np.array(pred_args)
            pred_args_arr=pred_args_arr.reshape(1,-1)
            model_prediction =model.predict(pred_args_arr)
            model_prediction=round(float(model_prediction),1)
            
        except ValueError:
            return "Please check if the values are entered correctly" 
    return render_template('index.html',prediction_text='Predicted Crop = {}'.format(model_prediction))
            
     
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
