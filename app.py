from flask import Flask,request,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('TreeCredz.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    data = request.get_json(force=True)
    input_json=[[data['steps']]]
    prediction = model.predict(input_json[0][0])
    
    output_res={
            "pred" : prediction[0]
        }
    
    return output_res
    
if __name__ == "__main__":
    app.run(debug=True)