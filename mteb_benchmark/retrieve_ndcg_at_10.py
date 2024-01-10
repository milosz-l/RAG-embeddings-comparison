import os
import json

folder_path = "results/pl/silver-retriever-base-v1.1/"

ndcg_at_10_values = []

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        with open(os.path.join(folder_path, filename), "r") as file:
            data = json.load(file)
            ndcg_at_10_values.append((filename, data["test"]["ndcg_at_10"]))

# Calculate average
average_ndcg_at_10 = sum(value for _, value in ndcg_at_10_values) / len(
    ndcg_at_10_values
)

ndcg_at_10_values.append(("average", average_ndcg_at_10))

for line in ndcg_at_10_values:
    print(line[0])
    print(round(line[1] * 100, 2))
