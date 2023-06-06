from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from main import chat_with_bot
import re
# from random import randint

app = Flask(__name__)
api = Api(app)

class botalent(Resource):
    def post(self):
        selectInput = request.form['select']
        chatInput = request.form['chatInput']

        if selectInput == "0":
            resp, pres, maks = chat_with_bot(chatInput)
            list_job = []
            highListJob = []
            ERROR_THRESHOLD = 20.0
            for ada in range(maks):
                if pres[ada] >= ERROR_THRESHOLD:
                    if resp[ada] == "Relating" or resp[ada] == "relating" or resp[ada] == "ating":
                        continue
                    elif resp[ada] == "Thinking" or resp[ada] == "thinking" or resp[ada] == "king":
                        continue
                    elif resp[ada] == "Impacting":
                        continue
                    elif resp[ada] == "Striving" or resp[ada] == "ving":
                        continue
                    else:
                        list_job.append("Skill: {}, {:0.2f}%".format(resp[ada], pres[ada]))

            sorting = sorted(list_job, key=lambda s: float(re.search(r'(\d+)\.', s).groups()[0]))

            for i in range(1, 6):
                # bil = randint(0, 4)
                # highListJob.append(sorting[(-i)-bil])
                highListJob.append(sorting[-i])

            data = '\n'.join(map(str, highListJob))
            return jsonify(chatBotReply="Rekomendasi :\n{}".format(data))

        elif selectInput == "1":
            resp, pres, maks = chat_with_bot(chatInput)
            list_job = []
            highListJob = []
            ERROR_THRESHOLD = 3

            for ada in range(maks):
                if pres[ada] >= ERROR_THRESHOLD:
                    if resp[ada] == "Relating" or resp[ada] == "relating" or resp[ada] == "ating":
                        continue
                    elif resp[ada] == "Thinking" or resp[ada] == "thinking" or resp[ada] == "king":
                        continue
                    elif resp[ada] == "Impacting":
                        continue
                    elif resp[ada] == "Striving" or resp[ada] == "ving":
                        continue
                    else:
                        list_job.append("Skill: {}, {:0.2f}%".format(resp[ada], pres[ada]))

            sorting = sorted(list_job, key=lambda s: float(re.search(r'(\d+)\.', s).groups()[0]))

            for i in range(1, 6):
                # bil = randint(0, 4)
                # highListJob.append(sorting[i+bil])
                highListJob.append(sorting[-i])

            data = '\n'.join(map(str, highListJob))
            return jsonify(chatBotReply="Tidak Rekomendasi :\n{}".format(data))

        else:
            resp, pres, maks = chat_with_bot(chatInput)
            mapping = []
            highDominan = []

            for ada in range(maks):
                if resp[ada] == "Relating":
                    mapping.append("Dominan : {}(sosialisasi), {:0.2f}%".format(resp[ada], pres[ada]))
                if resp[ada] == "Thinking":
                    mapping.append("Dominan : {}(Pemikir), {:0.2f}%".format(resp[ada], pres[ada]))
                if resp[ada] == "Impacting":
                    mapping.append("Dominan : {}(Pemimpin), {:0.2f}%".format(resp[ada], pres[ada]))
                if resp[ada] == "Striving":
                    mapping.append("Dominan : {}(Pekerja Keras), {:0.2f}%".format(resp[ada], pres[ada]))

            dominanSort = sorted(mapping, key=lambda s: float(re.search(r'(\d+)\.', s).groups()[0]))
            for i in range(1, 2):
                highDominan.append(dominanSort[-i])

            dataa = '\n'.join(map(str, highDominan))
            return jsonify(chatBotReply=dataa)


api.add_resource(botalent, "/chat", methods=["POST", "GET"])

if __name__ == '__main__':
    # app.run(host='192.168.55.18', port=5050, debug=True)
    app.run(host='192.168.18.20', port=5050, debug=True)
