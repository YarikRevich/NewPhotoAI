from logic.proto import api_pb2_grpc
from logic.proto import api_pb2
from logic.ai.fitted_model import FittedModel
from logic.ai.stock_model import create_stock_model

import grpc

class Tag(api_pb2_grpc.TagServicer):
    #Tag service handler
    #"IsHuman" method handles the request to check if there is a man in the image

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = FittedModel(create_stock_model())

    def RecognizeObject(self, request, context):
        tags = self.model.recognize(request.photo)
        return api_pb2.RecognizeObjectResponse(tags=tags)
