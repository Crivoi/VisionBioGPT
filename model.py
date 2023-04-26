from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataset import mimic_dataset
from preprocessing import preprocess_noteevent
from utils import ModelCheckpoint

model_checkpoint = ModelCheckpoint.BioGPT.value
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

if __name__ == '__main__':
    # set_seed(SEED)
    batch = mimic_dataset[0]
    prompt = f"{preprocess_noteevent(batch[0][0])}\n\nThe relevant diagnosis ICD code is"
    print(prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids=input_ids, max_length=1000, do_sample=True)
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_str)
