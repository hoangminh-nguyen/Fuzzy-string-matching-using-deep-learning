from flask import Flask
from flask import jsonify
from flask import request
from DeezyMatch import candidate_ranker

app = Flask(__name__)

@app.route('/search', methods=['POST'])
def search():
    # request_body = {"name": "@name_value"}
    request_body = request.get_json()
    search_name = request_body["name"]


    candidates_pd = \
        candidate_ranker(candidate_scenario="./combined/candidates_test",
                query=[search_name],
                ranking_metric="conf", 
                selection_threshold=0.8, 
                num_candidates=20, 
                search_size=100, 
                output_path="ranker_results/test_candidates_deezymatch_on_the_fly", 
                pretrained_model_path="./models/finetuned_trans_09052022/finetuned_trans_09052022.model", 
                pretrained_vocab_path="./models/finetuned_trans_09052022/finetuned_trans_09052022.vocab", 
                number_test_rows=20)

    return jsonify(candidates_pd["pred_score"])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)