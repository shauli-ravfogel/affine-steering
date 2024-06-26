python3 encode.py --model meta-llama/Llama-2-7b-chat-hf --layer -1 --pooling last --data tweets --batch_size 32
python3 encode.py --model google-bert/bert-base-uncased --layer -1 --pooling cls --data bios --batch_size 512
python3 encode.py --model meta-llama/Llama-2-7b-chat-hf --layer -1 --pooling last --data bios --batch_size 32
python3 encode.py --model openai-community/gpt2 --layer -1 --pooling last --data bios --batch_size 256

python3 classify_bios.py --model bert-base-uncased --layer last --pooling cls --do_pca 0
python3 classify_bios.py --model gpt2 --layer last --pooling last --do_pca 0
python3 classify_bios.py --model Llama-2-7b-chat-hf --layer last --pooling last --do_pca 1

