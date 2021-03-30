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

    def IsHuman(self, request, context):
        is_human = self.model.is_human(request.photo)
        return api_pb2.IsHumanResponse(ok=is_human)

    def IsDog(self, request, context):
        is_dog = self.model.is_dog(request.photo)
        return api_pb2.IsDogResponse(ok=is_dog)

