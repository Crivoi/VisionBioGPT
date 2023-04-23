from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

from utils import ModelCheckpoint, SEED

model_checkpoint = ModelCheckpoint.BioGPT.value
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

if __name__ == '__main__':
    set_seed(SEED)

    prompt = "Patient has been diagnosed with pneumonia."

    input_ids = tokenizer.encode(prompt, padding="max_length", truncation=True, return_tensors="pt")

    outputs = model(input_ids=input_ids)
    predicted_class = outputs.logits.argmax(dim=1).item()
    print(predicted_class)

    # # Generate the ICD codes
    # outputs = model.generate(input_ids=input_ids, max_length=50, do_sample=True)

    # input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # output = model.generate(input_ids=input_ids, max_length=1000, do_sample=True)
    # output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(output_str)
