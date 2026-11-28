"""Download only hash-pinned tokenizer JSON, and encode reference cases."""
import hashlib
import json
import pathlib
import sys
import unicodedata
import urllib.request

from tokenizers import Tokenizer

MODELS = {
    "openai-community/gpt2": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b",
    "Qwen/Qwen3-0.6B": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "unsloth/gemma-3-1b-it": "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
    "mistralai/Mistral-7B-v0.3": "e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49",
}
TEXTS = [
    "Hello, world!", "  leading  and trailing  ", "中文、日本語、한국어",
    "العربية", "👩‍💻👨‍👩‍👧‍👦", "def f():\n\treturn 42\r\n",
    " \t\n\n\t ", "cafe\u0301", "ÿ\x00\u0080", "abc <x> xyz",
]
root = pathlib.Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
cases = []
for repo, sha in MODELS.items():
    folder = root / repo.replace("/", "--")
    folder.mkdir(exist_ok=True)
    p = folder / "tokenizer.json"
    if not p.exists():
        with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/main/tokenizer.json", timeout=90) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != sha:
            raise RuntimeError(f"tokenizer changed: {repo}")
        p.write_bytes(data)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, repo
    tok = Tokenizer.from_file(str(p))
    model_cases = []
    for text in TEXTS:
        ids = tok.encode(text, add_special_tokens=False).ids
        expected = unicodedata.normalize("NFC", text) if repo.startswith("Qwen/") else text
        decoded = tok.decode(ids, skip_special_tokens=False)
        # Legacy Strip removes one leading space, even if it was supplied
        # by the input rather than by the tokenizer's dummy-prefix scheme.
        assert decoded == expected or (repo.startswith("mistralai/") and expected.startswith(" ") and decoded == expected[1:]), (repo, repr(text), repr(decoded))
        model_cases.append({"ids": ids, "expected": expected})
    cases.append({"model": repo, "file": str(p.relative_to(root)), "dummy_prefix": repo.startswith("mistralai/"), "cases": model_cases})
    print(repo, "reference cases OK", flush=True)
(root / "cases.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf8")
