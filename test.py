from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# Load the fine-tuned model and tokenizer
model_path = "./code_switch_mlm/checkpoint-50"
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")  # Load tokenizer from original model
model = AutoModelForMaskedLM.from_pretrained(model_path)

# Create a fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

test_examples = [
    "<mask>, kya scene hai?",                             # informal mix
    "Aaj ka din full <mask> hai.",                         # mix of Hindi and English adjective
    "Project pe <mask> progress chal raha hai.",           # using English word "progress"
    "Meri meeting <mask> ho gayi, totally.",               # extra English adverb
    "Tum <mask> kaam kar rahe ho, yaar?",                   # informal tone with code-switch
    "Yaar, kal ka schedule <mask> hai, let's plan properly.",  # adding English clause
    "Office mein <mask> mood hai aaj, it's really busy.",   # mixing expressions
    "Dinner ke baad, <mask>movie dekhne ka plan hai?",     # casual mix
    "College ke assignment ke liye <mask> study session ho raha hai.", # educational context
    "Bhai, aaj ka game <mask> exciting hai, must watch!"    # sport/entertainment mix
]


# use test_examples 
# for test_example in test_examples:
#     result = fill_mask(test_example)
#     print(f"\nTest Example: {test_example}")
#     print(result)

# Test with a simple spanish sentence.
test_sentence4 = "Bhai, <mask> was working on this project."
result4 = fill_mask(test_sentence4)
print(f"\nTest Sentence 4: {test_sentence4}")
for i, res in enumerate(result4):
    print(f"Option {i+1}:")
    for r in res:
        print(f"  Sequence: {r['sequence']}")
        print(f"  Score: {r['score']:.4f}")
        print(f"  Token: {r['token']}")
        print(f"  Token String: {r['token_str']}")
