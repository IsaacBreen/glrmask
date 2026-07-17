"""Incremental GPT-2 generation with a GLRMask JSON Schema constraint."""

from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import glrmask


MODEL_ID = "openai-community/gpt2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

vocab = glrmask.Vocab.from_id_to_bytes(
    {
        token_id: bytes(tokenizer.byte_decoder[c] for c in token)
        for token, token_id in tokenizer.get_vocab().items()
    }
)

schema = r'''
{
  "type": "object",
  "properties": {
    "sentiment": {"enum": ["positive", "negative", "neutral"]}
  },
  "required": ["sentiment"],
  "additionalProperties": false
}
'''

constraint = glrmask.Constraint.from_json_schema(schema, vocab)
state = constraint.start()

prompt = """Classify the sentiment of this review:

The performances were excellent, but the story dragged badly.

Return only a JSON object with a sentiment field."""

model_input = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
past_key_values = None
generated = []


@torch.inference_mode()
def model_step(input_ids, cache):
    return model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
    )


with ThreadPoolExecutor(max_workers=1) as executor:
    for _ in range(64):
        model_future = executor.submit(
            model_step,
            model_input,
            past_key_values,
        )
        allowed = torch.from_numpy(state.mask()).to(DEVICE)

        output = model_future.result()
        logits = output.logits[0, -1].float()
        logits.masked_fill_(~allowed, -torch.inf)

        if torch.isneginf(logits).all():
            raise RuntimeError("the constraint rejected every token")

        token_id = torch.multinomial(
            torch.softmax(logits / 0.8, dim=-1),
            num_samples=1,
        )
        token = int(token_id)

        generated.append(token)
        state.commit_token(token)

        model_input = token_id.view(1, 1)
        past_key_values = output.past_key_values

        if token == tokenizer.eos_token_id:
            break
    else:
        raise RuntimeError("generation did not finish within 64 tokens")

print(tokenizer.decode(generated, skip_special_tokens=True))
