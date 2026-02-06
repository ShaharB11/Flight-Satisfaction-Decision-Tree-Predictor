import csv
import math
import random

# Global list to map feature indices to readable names for printing
FEATURE_NAMES = [
    "Gender", "Customer Type", "Age", "Type of Travel", "Class", "Flight Distance",
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
    "Departure Delay", "Arrival Delay"
]


def bucket_age(age_str):
    try:
        age = float(age_str)
    except ValueError:
        return 0
    # Grouping continuous age into logical life stages
    if age <= 18: return 0  # 0-18 (Child)
    if age <= 35: return 1  # 19-35 (Young Adult)
    if age <= 60: return 2  # 36-60 (Adult)
    return 3  # 61+ (Senior)


def bucket_distance(dist_str):
    try:
        dist = float(dist_str)
    except ValueError:
        return 0
    # Categorizing flight distance into short/medium/long
    if dist <= 500: return 0
    if dist <= 1500: return 1
    if dist <= 3000: return 2
    return 3


def bucket_delay(delay_str):
    try:
        delay = float(delay_str)
    except ValueError:
        return 0
    # Converting minutes of delay into severity levels
    if delay <= 5: return 0  # Negligible delay
    if delay <= 30: return 1  # Moderate delay
    return 2  # Significant delay


def load_data(filename):
    """
    Loads and processes the CSV file.
    Converts categorical strings to integers and buckets continuous variables.
    NOW ROBUST: Skips rows with unknown categories instead of guessing.
    """
    data = []

    # Mappings to convert string categories into numerical IDs
    gender_map = {'Male': 0, 'Female': 1}
    cust_map = {'disloyal Customer': 0, 'Loyal Customer': 1}
    travel_map = {'Personal Travel': 0, 'Business travel': 1}
    class_map = {'Eco': 0, 'Eco Plus': 1, 'Business': 2}
    sat_map = {'neutral or dissatisfied': 0, 'satisfied': 1}

    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)  # Skip the header row
            except StopIteration:
                return []

            for row in reader:
                # Basic validation
                if not row or len(row) < 25:
                    continue

                try:
                    processed_row = [
                        gender_map[row[2]],  # Strict lookup
                        cust_map[row[3]],  # Strict lookup
                        bucket_age(row[4]),
                        travel_map[row[5]],  # Strict lookup
                        class_map[row[6]],  # Strict lookup
                        bucket_distance(row[7]),
                        int(row[8]), int(row[9]), int(row[10]), int(row[11]),
                        int(row[12]), int(row[13]), int(row[14]), int(row[15]),
                        int(row[16]), int(row[17]), int(row[18]), int(row[19]),
                        int(row[20]), int(row[21]),
                        bucket_delay(row[22]),
                        bucket_delay(row[23]),
                        sat_map[row[24]]  # Strict lookup for target too
                    ]
                    data.append(processed_row)
                except (ValueError, KeyError):
                    continue

    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return []

    return data








def entropy(data):
    """Calculates the entropy of a target label set."""
    if not data:
        return 0

    # Count occurrences of 0 and 1
    count_0 = 0
    count_1 = 0
    for value in data:
        if value == 0:
            count_0 += 1
        else:
            count_1 += 1

    total = len(data)
    # Calculate probabilities
    p_0 = count_0 / total
    p_1 = count_1 / total

    # If all examples are the same ,entropy is 0 (pure node)
    if p_0 == 0 or p_1 == 0:
        return 0

    # Standard Entropy formula: -Sum(p * log2(p))
    return -p_0 * math.log2(p_0) - p_1 * math.log2(p_1)


def calculate_chi_square(parent_data, left_data, right_data):
    """
    Calculates the Chi-Square statistic to determine if a split is statistically significant
    or just due to random chance (noise).
    """
    # 1. Analyze parent distribution (expected probability)
    parent_ones=0
    for row in parent_data:
        parent_ones += row[-1]
    parent_zeros = len(parent_data) - parent_ones

    if len(parent_data) == 0:
        return 0

    p_ones = parent_ones / len(parent_data)
    p_zeros = parent_zeros / len(parent_data)

    chi_sq = 0

    # 2. Compare actual child distribution vs expected distribution based on parent
    for child in [left_data, right_data]:
        total_child = len(child)
        if total_child == 0:
            continue

        # Expected counts if the split was purely random
        expected_ones = total_child * p_ones
        expected_zeros = total_child * p_zeros

        # Actual observed counts in the child node
        actual_ones=0
        for row in child:
            actual_ones += row[-1]
        actual_zeros = total_child - actual_ones

        # Chi-Square formula: Sum((Observed - Expected)^2 / Expected)
        if expected_ones > 0:
            chi_sq += ((actual_ones - expected_ones) ** 2) / expected_ones
        if expected_zeros > 0:
            chi_sq += ((actual_zeros - expected_zeros) ** 2) / expected_zeros

    return chi_sq



class DecisionNode:
    """
        Represents a single node in the Decision Tree.
        - If 'value' is not None, it is a Leaf Node (holds the final prediction).
        - If 'value' is None, it is an Internal Node (holds a question/split condition).
        """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # The column index used for the split
        self.threshold = threshold  # The value used to split (<= threshold vs > threshold)
        self.left = left  # Child node for 'True' path
        self.right = right  # Child node for 'False' path
        self.value = value  # Leaf value (0 or 1), None if internal node


# Global variable
MY_TREE = None


def split_data(data, feature_index, threshold):
    """Splits dataset into two lists based on a feature threshold."""
    left = []
    right = []
    for row in data:
        if row[feature_index] <= threshold:
            left.append(row)
        else:
            right.append(row)
    return left, right


def compute_gain(parent, l_child, r_child):
    """
    Calculates Information Gain
    Formula: Parent_Entropy - (Weighted_Average_Children_Entropy)
    """
    # 1. Create list of labels for parent
    parent_labels = []
    for row in parent:
        parent_labels.append(row[-1])

    # 2. Create list of labels for left child
    l_labels = []
    for row in l_child:
        l_labels.append(row[-1])

    # 3. Create list of labels for right child
    r_labels = []
    for row in r_child:
        r_labels.append(row[-1])

    # 4. Calculate individual entropies
    parent_ent = entropy(parent_labels)
    left_ent = entropy(l_labels)
    right_ent = entropy(r_labels)

    # 5. Calculate weights (portion of data in each side)
    weight_left = len(l_child) / len(parent)
    weight_right = len(r_child) / len(parent)

    # 6. Calculate final gain
    # Gain = Entropy(Parent) - [ (Weight_L * Entropy_L) + (Weight_R * Entropy_R) ]
    weighted_child_ent = (weight_left * left_ent) + (weight_right * right_ent)
    return parent_ent - weighted_child_ent




def get_best_split(data, features_indices):
    """Iterates over all features and values to find the split that maximizes Information Gain."""
    best_gain = -1
    best_feature = None
    best_threshold = None

    # Iterate over every available feature in the dataset
    for feature_index in features_indices:

        # Collect all unique values present in this column
        # We use a set to avoid checking the same threshold multiple times
        unique_values = set()
        for row in data:
            val = row[feature_index]
            unique_values.add(val)  # 'add' handles duplicates automatically

        # Try splitting the data based on each unique value
        for t in unique_values:

            # Split the dataset into two groups based on the current threshold 't'
            left, right = split_data(data, feature_index, t)

            # Optimization: Skip this split if it doesn't actually divide the data
            # (e.g., if everyone went to the left side, we learned nothing)
            if not left or not right:
                continue

            # Calculate how much 'Information Gain' this split provides
            gain = compute_gain(data, left, right)

            # 3. Track the best split found so far
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = t
    return best_feature, best_threshold


def learn_tree_structure(data, depth):
    """Recursively builds the decision tree using Chi-Square pruning."""

    # Determine the majority class (for leaf nodes or fallback)
    ones = sum([row[-1] for row in data])
    zeros = len(data) - ones
    majority = 1 if ones > zeros else 0

    # Base Cases / Stopping Conditions:
    # - Empty data
    # - Max depth reached (safeguard against infinite recursion)
    # - Pure node (only 0s or only 1s left)
    if not data or depth >= 10 or ones == 0 or zeros == 0:
        return DecisionNode(value=majority)

    # Find the best question to ask
    features = list(range(len(data[0]) - 1))
    best_feature, best_val = get_best_split(data, features)

    # If no split improves entropy, stop and return majority
    if best_feature is None:
        return DecisionNode(value=majority)

    left_data, right_data = split_data(data, best_feature, best_val)

    # Pruning Step: Chi-Square Test
    # Critical value 3.841 corresponds to alpha=0.05 for 1 degree of freedom.
    # If chi_square is low, the split is likely noise  Prune it (make it a leaf).
    if calculate_chi_square(data, left_data, right_data) < 3.841:
        return DecisionNode(value=majority)

    # Recursion: Build left and right subtrees
    left_node = learn_tree_structure(left_data, depth + 1)
    right_node = learn_tree_structure(right_data, depth + 1)

    return DecisionNode(feature_index=best_feature, threshold=best_val, left=left_node, right=right_node)


def predict(node, row):
    """
        Takes a trained tree node and a data row.
        It travels down the tree (left or right) based on the row's values
        until it hits a Leaf Node, then returns the prediction (0 or 1).
        """
    while node.value is None:
        if row[node.feature_index] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value


def print_tree_structure(node, spacing=""):
    """Recursive print the tree structure visually."""
    if node is None: return

    # If leaf, print the prediction
    if node.value is not None:
        print(f"{spacing}Leaf: Predict {node.value}")
        return

    #Try to use the readable name if available
    feature_name = f"Feature {node.feature_index}"
    if 0 <= node.feature_index < len(FEATURE_NAMES):
        feature_name = FEATURE_NAMES[node.feature_index]

    print(f"{spacing}{feature_name} <= {node.threshold}?")
    print(f"{spacing}--> True:")
    print_tree_structure(node.left, spacing + "  ")
    print(f"{spacing}--> False:")
    print_tree_structure(node.right, spacing + "  ")



def build_tree(ratio):
    global MY_TREE
    all_data = load_data('flights.csv')
    # Shuffling ensures the split is random and representative
    random.shuffle(all_data)
    split_idx = int(len(all_data) * ratio)

    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    # Build tree on the training portion
    MY_TREE = learn_tree_structure(train_data, depth=0)

    # Print the decision tree structure
    print("--- Decision Tree Structure ---")
    print_tree_structure(MY_TREE)

    # Report the error on the remaining data (test set)
    if len(test_data) > 0:
        mistakes = 0
        for row in test_data:
            # Compare prediction vs actual label (last column)
            if predict(MY_TREE, row) != row[-1]:
                mistakes += 1
        error_rate = mistakes / len(test_data)
        print(f"\nError on validation set (Ratio {ratio}): {error_rate:.4f}")
    else:
        print("\nNo data left for validation (Ratio=1.0).")


def tree_error(k):
    data = load_data('flights.csv')
    random.shuffle(data)

    n = len(data)
    fold_size = n // k
    total_error = 0

    # K-Fold Cross Validation Loop
    for i in range(k):
        # Determine start and end indices for the 'Validation' fold
        start = i * fold_size
        # Handle the last fold carefully to include any remainder rows
        end = (i + 1) * fold_size if i < k - 1 else n

        # Slicing the data:
        # validation_set is the current "slice" being tested
        # train_set is everything else concatenated
        validation_set = data[start:end]
        train_set = data[:start] + data[end:]

        # Build a temporary tree for this specific fold (does not overwrite global MY_TREE)
        temp_tree = learn_tree_structure(train_set, depth=0)

        mistakes = 0
        for row in validation_set:
            if predict(temp_tree, row) != row[-1]:
                mistakes += 1

        fold_err = mistakes / len(validation_set)
        total_error += fold_err

    # Return the average error across all k folds
    avg_error = total_error / k
    return avg_error


def will_be_satisfied(row_input):
    global MY_TREE

    # If the global tree hasn't been built yet (e.g. build_tree wasn't called),
    # build it now.
    if MY_TREE is None:
        full_data = load_data('flights.csv')
        MY_TREE = learn_tree_structure(full_data, depth=0)
    return predict(MY_TREE, row_input)


