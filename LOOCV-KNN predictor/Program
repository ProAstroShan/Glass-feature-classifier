# Import dependent modules
import numpy as np
import collections
from itertools import combinations

try:
    raw_data = open("glass1.csv", "r")
except FileNotFoundError as e:
    print(e)
else:
    data_list: list[str] = raw_data.readlines()

    formatted_data: list[list[str]] = [row[:-1].split(",") for row in data_list]

# feature headers
column_headers: list[str] = formatted_data[:1][0]

# split features and labels
unformatted_training_data: list[list[str]] = []
glass_type_data: list[str] = []

for sample_row in formatted_data[1:]:
    glass_type_data.append(sample_row.pop())
    unformatted_training_data.append(sample_row)

# number of features (excluding label)
num_features = len(column_headers) - 1
feature_names = column_headers[:-1]

# store final results
feature_results = []

# loop over ALL feature pairs
for f1, f2 in combinations(range(num_features), 2):

    # build 2-feature dataset
    training_data_pair = []

    for sample in unformatted_training_data:
        training_data_pair.append([float(sample[f1]), float(sample[f2])])

    training_data_pair = np.array(training_data_pair)

    # z-score normalisation (per feature column)
    normalized_pair = []

    for col in range(2):
        col_data = training_data_pair[:, col]
        mean = np.mean(col_data)
        std = np.std(col_data)
        normalized_pair.append(((col_data - mean) / std).tolist())

    training_data_formatted = np.array(normalized_pair).T.tolist()

    # LOOCV predictions
    predictions = []

    for i in range(len(training_data_formatted)):
        test_data = training_data_formatted[i]
        actual_glass_type = glass_type_data[i]

        current_training_set = (training_data_formatted[:i] + training_data_formatted[i + 1 :])
        current_training_glass_types = glass_type_data[:i] + glass_type_data[i + 1 :]

        distances_to_training_samples = []

        for j, train_sample in enumerate(current_training_set):
            temp_distance = 0

            for k in range(len(test_data)):
                temp_distance += (train_sample[k] - test_data[k]) ** 2

            distances_to_training_samples.append((temp_distance**0.5, current_training_glass_types[j]))

        k = 3
        top_k_neighbors = sorted(distances_to_training_samples)[:k]

        neighbor_glass_types = [neighbor[1] for neighbor in top_k_neighbors]

        predicted_glass_type = collections.Counter(neighbor_glass_types).most_common(1)[0][0]

        predictions.append((actual_glass_type, predicted_glass_type))

    # accuracy
    correct_predictions = sum(1 for actual, predicted in predictions if actual == predicted)
    accuracy = correct_predictions / len(predictions)

    feature_results.append((feature_names[f1], feature_names[f2], accuracy))

# sort results
feature_results.sort(key=lambda x: x[2], reverse=True)

print("\n--- Feature Pair Rankings ---")
for f1, f2, acc in feature_results:
    print(f"{f1} + {f2}: {acc:.2%}")
