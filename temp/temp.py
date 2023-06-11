from transformers import BioGptForCausalLM

import settings

if __name__ == '__main__':
    model = BioGptForCausalLM.from_pretrained(settings.BIOGPT_CHECKPOINT)
    pass
