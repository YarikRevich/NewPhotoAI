# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: logic/proto/api.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='logic/proto/api.proto',
  package='main',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x15logic/proto/api.proto\x12\x04main\"\x1f\n\x0eIsHumanRequest\x12\r\n\x05photo\x18\x01 \x01(\x0c\"\x1d\n\x0fIsHumanResponse\x12\n\n\x02ok\x18\x01 \x01(\x08\"\x1d\n\x0cIsDogRequest\x12\r\n\x05photo\x18\x01 \x01(\x0c\"\x1b\n\rIsDogResponse\x12\n\n\x02ok\x18\x01 \x01(\x08\x32s\n\x03Tag\x12\x38\n\x07IsHuman\x12\x14.main.IsHumanRequest\x1a\x15.main.IsHumanResponse\"\x00\x12\x32\n\x05IsDog\x12\x12.main.IsDogRequest\x1a\x13.main.IsDogResponse\"\x00\x62\x06proto3'
)




_ISHUMANREQUEST = _descriptor.Descriptor(
  name='IsHumanRequest',
  full_name='main.IsHumanRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='photo', full_name='main.IsHumanRequest.photo', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=31,
  serialized_end=62,
)


_ISHUMANRESPONSE = _descriptor.Descriptor(
  name='IsHumanResponse',
  full_name='main.IsHumanResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='ok', full_name='main.IsHumanResponse.ok', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=64,
  serialized_end=93,
)


_ISDOGREQUEST = _descriptor.Descriptor(
  name='IsDogRequest',
  full_name='main.IsDogRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='photo', full_name='main.IsDogRequest.photo', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=95,
  serialized_end=124,
)


_ISDOGRESPONSE = _descriptor.Descriptor(
  name='IsDogResponse',
  full_name='main.IsDogResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='ok', full_name='main.IsDogResponse.ok', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=126,
  serialized_end=153,
)

DESCRIPTOR.message_types_by_name['IsHumanRequest'] = _ISHUMANREQUEST
DESCRIPTOR.message_types_by_name['IsHumanResponse'] = _ISHUMANRESPONSE
DESCRIPTOR.message_types_by_name['IsDogRequest'] = _ISDOGREQUEST
DESCRIPTOR.message_types_by_name['IsDogResponse'] = _ISDOGRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

IsHumanRequest = _reflection.GeneratedProtocolMessageType('IsHumanRequest', (_message.Message,), {
  'DESCRIPTOR' : _ISHUMANREQUEST,
  '__module__' : 'logic.proto.api_pb2'
  # @@protoc_insertion_point(class_scope:main.IsHumanRequest)
  })
_sym_db.RegisterMessage(IsHumanRequest)

IsHumanResponse = _reflection.GeneratedProtocolMessageType('IsHumanResponse', (_message.Message,), {
  'DESCRIPTOR' : _ISHUMANRESPONSE,
  '__module__' : 'logic.proto.api_pb2'
  # @@protoc_insertion_point(class_scope:main.IsHumanResponse)
  })
_sym_db.RegisterMessage(IsHumanResponse)

IsDogRequest = _reflection.GeneratedProtocolMessageType('IsDogRequest', (_message.Message,), {
  'DESCRIPTOR' : _ISDOGREQUEST,
  '__module__' : 'logic.proto.api_pb2'
  # @@protoc_insertion_point(class_scope:main.IsDogRequest)
  })
_sym_db.RegisterMessage(IsDogRequest)

IsDogResponse = _reflection.GeneratedProtocolMessageType('IsDogResponse', (_message.Message,), {
  'DESCRIPTOR' : _ISDOGRESPONSE,
  '__module__' : 'logic.proto.api_pb2'
  # @@protoc_insertion_point(class_scope:main.IsDogResponse)
  })
_sym_db.RegisterMessage(IsDogResponse)



_TAG = _descriptor.ServiceDescriptor(
  name='Tag',
  full_name='main.Tag',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=155,
  serialized_end=270,
  methods=[
  _descriptor.MethodDescriptor(
    name='IsHuman',
    full_name='main.Tag.IsHuman',
    index=0,
    containing_service=None,
    input_type=_ISHUMANREQUEST,
    output_type=_ISHUMANRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='IsDog',
    full_name='main.Tag.IsDog',
    index=1,
    containing_service=None,
    input_type=_ISDOGREQUEST,
    output_type=_ISDOGRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_TAG)

DESCRIPTOR.services_by_name['Tag'] = _TAG

# @@protoc_insertion_point(module_scope)
