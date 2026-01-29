from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    print("Loading model 'prithivida/informal_to_formal_styletransfer'...")
    tokenizer = AutoTokenizer.from_pretrained("prithivida/informal_to_formal_styletransfer")
    model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/informal_to_formal_styletransfer")
    
    print("Model loaded successfully!\n")

    examples = [
        "u wanna go to the movies tonight?",
        "idk what u mean by that lol",
        "thats gotta be the craziest thing ive ever seen"
    ]

    print("--- Running Examples ---")
    for text in examples:
        inputs = tokenizer.encode("informal: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
        formal_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Informal: {text}")
        print(f"Formal:   {formal_text}\n")

    print("--- Interactive Mode ---")
    print("Enter your own informal text (or type 'exit' to quit):")
    while True:
        text = input("> ")
        if text.lower() in ('exit', 'quit'):
            break
        
        inputs = tokenizer.encode("informal: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
        formal_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Formal:   {formal_text}\n")

if __name__ == "__main__":
    main()
