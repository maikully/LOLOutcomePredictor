from flask import Flask, request
import predictor

api = Flask(__name__)

total_champs = 162

predictor.main()


@api.route("/getPrediction", methods=["POST"])
def getPrediction():
  champs = request.args['data']
  encodings = [[name_to_encoding(x.lower().strip()) for x in champs]]
  predictor.make_one_hot(total_champs, encodings)
  response_body = {
      "name": "Nagato",
      "about": "Hello! I'm a full stack developer that loves python and javascript"
  }

  return response_body

