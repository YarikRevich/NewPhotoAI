from logic.proto import implementation
from logic.proto import api_pb2_grpc

from concurrent import futures
import grpc
import os

from termcolor import cprint


def start_server():

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=100),
        options=[
            ("grpc.max_send_message_length", 50000000),
            ("grpc.max_receive_message_length", 50000000)
        ]
    )

    api_pb2_grpc.add_TagServicer_to_server(implementation.Tag(), server)

    ai_addr = os.environ.get("aiAddr", lambda: (cprint("aiAddr is not written in credentials.sh file\n", color="red"), exit(0)))
    if isinstance(ai_addr, type(lambda: None)):
        ai_addr()

    server.add_insecure_port(ai_addr)

    server.start()

    server.wait_for_termination()
