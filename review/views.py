import json
import pickle

from django.http import HttpResponse
from rest_framework import status

import pandas as pd


def get_sentiment(request):
    result = {}
    if request.method == 'POST':
        body = json.loads(request.body)
        review = body['data']
        result = get_prediction(review)
    return HttpResponse(json.dumps(result, indent=2), content_type="application/json", status=status.HTTP_200_OK)


def get_prediction(text_list):
    model_path = 'static/review_ml_model/finalized_model.sav'    # need to change this
    if 'classifier' not in globals():
        classifier = pickle.load(open(model_path, 'rb'))
    text = pd.Series(text_list, name='text')
    sentiment = pd.Series(classifier.predict(text), name='tag_name')
    confidence = pd.Series(list(map(lambda x: round(
        max(x), 2), classifier.predict_proba(sentiment).tolist())), name='confidence')
    df = pd.concat([text, sentiment, confidence], axis=1)
    # to ensure expected format
    df['error'] = False # need to change this
    return df.to_dict('records')
