import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from flask import Flask, jsonify, request

marks_pipeline = joblib.load('./_marks_pred_pipeline_Comb_5std.pkl') # Path to model here

app = Flask(__name__)


@app.route('/predict/marks', methods=['POST', 'GET'])
def predict_marks():
  data = request.json
  features = data['features']
  feat = {
      "CERTIFICATEID": features[0],
      "age": features[1],
      "SCHOOLID": features[2],
      "SUBJECTNAME": features[3],
      "USER_GROUP": features[4],
      "Part1TheoryMarks": features[5],
      "Part1PracticalMarks": features[6],
      "Part1TotalAllSubsMarks": features[7],
      "SUBJECTTYPE": features[8],
      "MEDIUMTOUGHTTYPE": features[9],
      "Part1Marks": features[10]
  }
  feat = pd.DataFrame(feat, index=[0])
  prediction = float(marks_pipeline.predict(feat)[0])

  return jsonify({'Predicted marks': prediction * 100})


if __name__ == '__main__':
  from waitress import serve
  serve(app.run(host='0.0.0.0', port=5000, threaded=True))
