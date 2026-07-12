from collections import Counter

from HotpotQA_Distractor.data.load_data import load_hotpotqa


# CONFIG = "distractor"

# NUM_SAMPLES = 5000


# def analyze_split(split_name, dataset):

#     counter = Counter()
#     first_levels = []

#     for idx, sample in enumerate(dataset):

#         if idx >= NUM_SAMPLES:
#             break

#         level = sample["level"].lower()

#         counter[level] += 1

#         if idx < 100:
#             first_levels.append(level)

#     print("\n" + "=" * 80)
#     print(split_name.upper())
#     print("=" * 80)

#     total = sum(counter.values())

#     print(f"\nTotal checked: {total}")

#     print("\nLevel distribution:")

#     for level in ["easy", "medium", "hard"]:

#         count = counter[level]

#         pct = 100 * count / total

#         print(
#             f"{level:<6} : "
#             f"{count:>5} "
#             f"({pct:.2f}%)"
#         )

#     print("\nFirst 100 levels:")

#     print(first_levels)


# def main():

#     dataset = load_hotpotqa(CONFIG)

#     analyze_split(
#         "train",
#         dataset["train"]
#     )

#     analyze_split(
#         "validation",
#         dataset["validation"]
#     )


# if __name__ == "__main__":
#     main()

from collections import Counter

dataset = load_hotpotqa("distractor")

counter = Counter()

for sample in dataset["validation"]:

    counter[sample["level"]] += 1

print(counter)