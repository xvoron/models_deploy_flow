start:
	torchserve --start --model-store model_store --workflow-store wf_store --ncs --ts-config config.properties
build:
	torch-workflow-archiver -f --workflow-name test_wf \
		--spec-file workflow.yaml --handler workflow_handler.py --export-path wf_store

start_model:
	torchserve --start --model-store model_store --ncs --ts-config config.properties --models all
stop:
	torchserve --stop

register:
	curl -X POST  "http://localhost:8081/workflows?url=test_wf.war"

infer:
	curl -X POST "http://localhost:8080/predictions/test_wf" -T test.jpg

wf_infer:
	curl -X POST "http://localhost:8080/wfpredict/test_wf" -T test.jpg
