

build-litgpt:
	docker build -t litgpt:latest -f Dockerfile.litgpt .

run-litgpt:
	docker stop litgpt && docker rm litgpt || true
	docker run --runtime nvidia -it --restart always -d --network=host -v /checkpoints/litgpt:/checkpoints --name litgpt litgpt

build-xtts:
	docker build -t xtts:latest -f Dockerfile.xtts .

run-xtts:
	docker run --runtime nvidia -it --rm -d --network=host  -v /checkpoints/litgpt:/checkpoints --name xtts xtts

build-f5:
	docker build -t f5:latest -f Dockerfile.f5 .

run-f5:
	docker run --runtime nvidia -it --restart always -d --network=host -v /checkpoints/litgpt:/checkpoints --name f5 f5

build-melo:
	docker build -t melo:latest -f Dockerfile.melo .

run-melo:
	docker stop melo && docker rm melo || true
	docker run --runtime nvidia -it --restart always -d --network=host -v /checkpoints/litgpt:/checkpoints --name melo melo

build-fish:
	docker build -t fish:latest -f Dockerfile.fish .

run-fish:
	docker run --runtime nvidia -it --rm -d --network=host -v /checkpoints/litgpt:/checkpoints --name fish fish

build-whisper:
	docker build -t whisper:latest -f Dockerfile.whisper .

run-whisper:
	docker stop whisper && docker rm whisper || true
	docker run --runtime nvidia -it --restart always -d -v /checkpoints/whisper:/data/models/whisper --network=host --name whisper whisper:latest
