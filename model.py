from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM

if __name__ == '__main__':
    model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    set_seed(42)
    output = generator("", max_length=20, num_return_sequences=5, do_sample=True)
    for i in output:
        print(i.get('generated_text'))
