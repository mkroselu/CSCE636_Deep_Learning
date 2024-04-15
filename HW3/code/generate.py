import torch
from datasets import load_dataset
from tqdm import tqdm

from model import GPT, GPTConfig
from tokenizer import build_tokenizer


def load_model(model_path, config):
    model = GPT(config)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model


def generate_sample(model, tokenizer, conditions, max_length):
    model.eval()
    input_ids = tokenizer.generation_encode(conditions)
    input_ids = torch.tensor([input_ids], dtype=torch.long).cuda()
    len_conditions = len(input_ids[0])

    with torch.no_grad():
        for _ in range(max_length - len_conditions):
            # Generate one token at a time, and append it to the input to do generation iteratively until </s> is generated
            ### YOUR CODE HERE ###
            
            logits, _, _ = model(input_ids)

            # Get the logits for the last generated token
            logits_last_token = logits[:, -1, :]

            # Sample the next token using the logits
            next_token = torch.argmax(logits_last_token, dim=-1)

            # Append the next token to the input_ids for the next iteration
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=-1)

            # Check if </s> token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break 

            ### YOUR CODE HERE ###

    generated_text = tokenizer.decode(input_ids[0][len_conditions:])
    return generated_text


def generate(args):

    data_SCAN = load_dataset("scan", args.data_split)

    max_len = args.max_len
    tokenizer, vocab_size = build_tokenizer(args, data_SCAN, max_len, args.output_tokenizer_dir)

    mconf = GPTConfig(vocab_size, max_len,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      isconditional=True)

    # Load model and tokenizer
    print("loading model")
    model = load_model(args.ckpt_path, mconf).cuda()
    print('total params:', sum(p.numel() for p in model.parameters()))


    # Sample generation
    test_data = data_SCAN['test']
    correct_count = 0
    pbar = tqdm(enumerate(test_data), total=len(test_data))
    for i, data in pbar:
        generated_actions = generate_sample(model, tokenizer, data['commands'], max_len)
        if generated_actions == data['actions']:
            correct_count += 1
        pbar.set_description(f'Accuracy: {correct_count / (i + 1):.4f}')
    print(f'Test accuracy: {correct_count / len(test_data)}')
