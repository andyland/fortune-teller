

build-litgpt:
	docker build -t litgpt:latest -f litgpt/Dockerfile litgpt

run-litgpt:
	docker stop litgpt && docker rm litgpt || true
	docker run --runtime nvidia -it --restart always -d --network=host -v /checkpoints/litgpt:/checkpoints --name litgpt litgpt

build-assistant:
	docker build -t assistant:latest -f assistant/Dockerfile assistant

run-assistant:
	docker run --runtime nvidia -it --restart always -d --network=host -v /checkpoints/litgpt:/checkpoints --name assistant assistant

build-melo:
	docker build -t melo:latest -f melo/Dockerfile melo

run-melo:
	docker stop melo && docker rm melo || true
	docker run --runtime nvidia -it --restart always -d --network=host -v /checkpoints/litgpt:/checkpoints --name melo melo

build-whisper:
	docker build -t whisper:latest -f whisper/Dockerfile whisper

run-whisper:
	docker stop whisper && docker rm whisper || true
	docker run --runtime nvidia -it --restart always -d -v /checkpoints/whisper:/data/models/whisper --network=host --name whisper whisper:latest

build-silero:
	docker build -t silero:latest -f silero/Dockerfile silero

run-silero:
	docker stop silero && docker rm silero || true
	docker run  --runtime nvidia -it --restart always -d -v /checkpoints/torch:/data/models/torch  --network=host --name silero silero:latest
