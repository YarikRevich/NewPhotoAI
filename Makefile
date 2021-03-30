.PHONY: stub build proto

stub: 
	@echo "It is a stub"

proto:
	export PATH=$$PATH:$$GOPATH/bin;\
	python3 -m grpc_tools.protoc -I $$GOPATH/src/NewPhotoAI --python_out=. --grpc_python_out=. logic/proto/api.proto 