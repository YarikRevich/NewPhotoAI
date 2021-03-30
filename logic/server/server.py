from logic.proto import implementation
from logic.proto import api_pb2_grpc

from concurrent import futures
import grpc


def start_server():

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=100),
        options=[
            ("grpc.max_send_message_length", 50000000),
            ("grpc.max_receive_message_length", 50000000)
        ]
    )

    api_pb2_grpc.add_TagServicer_to_server(implementation.Tag(), server)

    server.add_insecure_port("[::]:9089",)

    server.start()

    server.wait_for_termination()
