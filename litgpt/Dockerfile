FROM dustynv/pytorch:2.6-r36.4.0-cu128-24.04

RUN pip install litgpt==0.5.9 litserve bitsandbytes

CMD ["litgpt", "serve", "/checkpoints/qwen", "--openai_spec", "true", "--quantize", "bnb.nf4", "--precision", "bf16-true"]
