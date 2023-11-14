import json

from django.http import JsonResponse
from django.views import View
from django.views.generic import TemplateView
from transformers import BioGptTokenizer, BioGptModel, BioGptForCausalLM, pipeline


class HomeView(TemplateView):
    template_name = 'index.html'


class ModelAPIView(View):
    model_checkpoint = "microsoft/biogpt"
    tokenizer: BioGptTokenizer = BioGptTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model: BioGptModel = BioGptForCausalLM.from_pretrained(model_checkpoint)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            prompt = data.get("prompt", "")
            max_length = int(data.get("max_length", 50))
            output = self.generator(prompt, max_length=max_length, num_return_sequences=1, do_sample=True)[0]
            return JsonResponse({'text': output.get('generated_text', 'An error occurred!')})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    def get(self, request, *args, **kwargs):
        return JsonResponse({"error": "Only POST requests are allowed"}, status=405)
