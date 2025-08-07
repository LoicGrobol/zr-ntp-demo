import json

for split in ["validation", "train", "test"]:
    with open(f"{split}.jsonl") as in_stream, open(f"{split}.txt", "w") as out_stream:
        for line in in_stream:
            d = json.loads(line)
            for m in d["messages"]:
                out_stream.write(m)
                out_stream.write("\n")
