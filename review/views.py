import json
import pickle

from django.http import HttpResponse
from rest_framework import status, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings
import pandas as pd


class GetSentiment(APIView):
    def post(self, request):
        body = request.data
        reviews = body['data']
        try:
            result = self.get_prediction(reviews)
        except Exception as e:
            result = []
            for review in reviews:
                result.append({'text': review, 'error': True, 'description': str(e)})

        return Response(result, status=status.HTTP_200_OK)

    def get_prediction(self, text_list):
        # model_path = 'static/review/finalized_model.sav'    # need to change this
        # if 'classifier' not in globals():
        #    classifier = pickle.load(open(model_path, 'rb'))
        classifier = settings.CLASSIFIER
        text = pd.Series(text_list, name='text')
        sentiment = pd.Series(classifier.predict(text), name='tag_name')
        confidence = pd.Series(list(map(lambda x: round(
            max(x), 2), classifier.predict_proba(sentiment).tolist())), name='confidence')
        df = pd.concat([text, sentiment, confidence], axis=1)
        # to ensure expected format
        df['error'] = False  # need to change this
        return df.to_dict('records')




