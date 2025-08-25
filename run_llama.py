import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_llama_3_2_1b():
    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision for efficiency
        # device_map="auto",          # Automatically handle device placement
    )

    # Set pad token if not already set - use a different token than eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    # Example prompt
    prompt = "The future of artificial intelligence is"

    print(f"Prompt: {prompt}")
    print("Generating response...")

    # Tokenize input with explicit attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,  # Explicitly pass attention mask
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")


if __name__ == "__main__":
    run_llama_3_2_1b()
