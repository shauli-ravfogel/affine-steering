python3 encode.py --model meta-llama/Llama-2-7b-chat-hf --layer -1 --pooling last --data tweets --batch_size 32
python3 encode.py --model google-bert/bert-base-uncased --layer -1 --pooling cls --data bios --batch_size 512
python3 encode.py --model meta-llama/Llama-2-7b-chat-hf --layer -1 --pooling last --data bios --batch_size 32
python3 encode.py --model openai-community/gpt2 --layer -1 --pooling last --data bios --batch_size 512


