# EXAMPLE: SAMPLES AND LEAVES IN A DECISION TREE

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)

print("="*80)
print("SMALL DATASET AND VISIBLE")
print("="*80)

# Only 12 people to see everything clearly
data = pd.DataFrame({
    'experience': [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13],
    'education_years': [12, 12, 16, 16, 16, 18, 18, 18, 18, 20, 20, 20],
    'salary_thousands': [30, 35, 45, 50, 70, 80, 85, 90, 95, 110, 115, 120]
})

print("\nFull Dataset (12 samples = 12 people):")
print(data)

print("\n" + "="*80)
print("KEY CONCEPT: WHAT IS A SAMPLE?")
print("="*80)

print("""
SAMPLE = 1 ROW = 1 OBSERVATION = 1 PERSON/HOUSE/PRODUCT

In our example:
  - Sample 1: Person with 1 year experience, 12 years education, salary $30k
  - Sample 2: Person with 2 years experience, 12 years education, salary $35k
  - ... (and so on)

Total: 12 SAMPLES (12 people in the dataset)

FEATURE = 1 COLUMN = 1 ATTRIBUTE
  - Feature 1: experience (values: 1, 2, 3, ...)
  - Feature 2: education_years (values: 12, 16, 18, 20)

Total: 2 FEATURES (2 columns that describe people)

SAMPLE ≠ FEATURE!
""")

# ============================================================================
print("\n" + "="*80)
print("STEP 2: SPLIT X (FEATURES) AND y (TARGET)")
print("="*80)

X = data.drop('salary_thousands', axis=1)
y = data['salary_thousands']

print("\nX (FEATURES) - Shape:", X.shape)
print("Interpretation: 12 samples × 2 features")
print(X)

print("\ny (TARGET) - Shape:", y.shape)
print("Interpretation: 12 salary values (one per each sample)")
print(y.values)

# ============================================================================
print("\n" + "="*80)
print("STEP 3: TRAIN SIMPLE TREE")
print("="*80)

tree_model = DecisionTreeRegressor(
    max_depth=2,              # Only 2 levels (root + 1 level)
    min_samples_split=3,      # Needs 3+ samples to split
    min_samples_leaf=2,       # Each leaf must have 2+ samples
    random_state=42
)

print("\nTREE CONFIGURATION:")
print(f"  max_depth = 2")
print(f"    → The tree can have a maximum depth of 2 levels")
print(f"    → Level 0 = Root, Level 1 = Intermediate nodes, Level 2 = Leaves")
print()
print(f"  min_samples_split = 3")
print(f"    → A node only splits if it has 3 or MORE samples")
print(f"    → If a node has 2 samples, it does NOT split (remains as a leaf)")
print()
print(f"  min_samples_leaf = 2")
print(f"    → Each final leaf must have MINIMUM 2 samples")
print(f"    → Leaves with only 1 sample are not allowed")

print("\nTraining tree...")
tree_model.fit(X, y)
print("✓ Tree trained")

# ============================================================================
print("\n" + "="*80)
print("STEP 4: ANALYZE TREE STRUCTURE")
print("="*80)

# Get tree information
n_nodes = tree_model.tree_.node_count
n_leaves = tree_model.tree_.n_leaves
depth = tree_model.tree_.max_depth

print(f"\nTREE STATISTICS:")
print(f"  Total nodes: {n_nodes}")
print(f"  Total leaves: {n_leaves}")
print(f"  Maximum depth: {depth}")

# Information for each node
print(f"\nNODE-BY-NODE BREAKDOWN:")
print("="*80)

tree = tree_model.tree_
feature_names = ['experience', 'education_years']

for i in range(n_nodes):
    # Check if it's a leaf
    is_leaf = tree.children_left[i] == tree.children_right[i]

    # Node information
    n_samples = tree.n_node_samples[i]
    predicted_value = tree.value[i][0][0]

    print(f"\nNODE {i}:")
    print(f"  Type: {' LEAF (final node)' if is_leaf else ' INTERNAL NODE (makes a question)'}")
    print(f"  Number of samples: {n_samples} people")
    print(f"  Average value: ${predicted_value:.2f}k")

    if not is_leaf:
        # It's a node that splits
        feature = tree.feature[i]
        threshold = tree.threshold[i]
        left_child = tree.children_left[i]
        right_child = tree.children_right[i]

        print(f"  Question: Is {feature_names[feature]} <= {threshold:.2f}?")
        print(f"    → YES: goes to Node {left_child}")
        print(f"    → NO: goes to Node {right_child}")
    else:
        print(f"   FINAL PREDICTION: ${predicted_value:.2f}k")
        print(f"  (This is the average of the {n_samples} samples that reached here)")

# ============================================================================
print("\n" + "="*80)
print("STEP 5: TRACE A SAMPLE THROUGH THE TREE")
print("="*80)

# Take the first person as an example
sample_example = X.iloc[0:1]
sample_features = X.iloc[0]
real_salary = y.iloc[0]

print("\n EXAMPLE PERSON (Sample #1):")
print(f"  Experience: {sample_features['experience']} years")
print(f"  Education: {sample_features['education_years']} years")
print(f"  Real salary: ${real_salary}k")

# Get the path this person takes in the tree
path = tree_model.decision_path(sample_example)
nodes_visited = path.indices

print(f"\nPATH IN THE TREE (visited {len(nodes_visited)} nodes):")

for i, node_id in enumerate(nodes_visited):
    is_leaf = tree.children_left[node_id] == tree.children_right[node_id]
    n_samples = tree.n_node_samples[node_id]
    value = tree.value[node_id][0][0]

    print(f"\n  Step {i+1} - NODE {node_id}:")
    print(f"    Type: {' LEAF' if is_leaf else ' NODE'}")
    print(f"    Samples here: {n_samples} people")
    print(f"    Average: ${value:.2f}k")

    if not is_leaf:
        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        person_value = sample_features[feature_names[feature]]

        answer = "YES" if person_value <= threshold else "NO"
        print(f"    Question: Is {feature_names[feature]} <= {threshold:.2f}?")
        print(f"    This person's value: {person_value}")
        print(f"    Answer: {answer}")
    else:
        print(f"     FINAL PREDICTION: ${value:.2f}k")

prediction = tree_model.predict(sample_example)[0]
print(f"\n RESULT:")
print(f"  Tree prediction: ${prediction:.2f}k")
print(f"  Real salary: ${real_salary}k")
print(f"  Error: ${abs(real_salary - prediction):.2f}k")

# ============================================================================
print("\n" + "="*80)
print("STEP 6: VISUALIZE WHICH SAMPLES ARE IN EACH LEAF")
print("="*80)

# Get which leaf each sample reaches
assigned_leaves = tree_model.apply(X)

print("\n SAMPLE ASSIGNMENT TO LEAVES:")
print("="*80)

# Create DataFrame with assignments
df_assignment = data.copy()
df_assignment['assigned_leaf'] = assigned_leaves
df_assignment['prediction'] = tree_model.predict(X)

print(df_assignment)

print("\nANALYSIS BY LEAF:")
print("="*80)

for leaf_id in df_assignment['assigned_leaf'].unique():
    samples_in_leaf = df_assignment[df_assignment['assigned_leaf'] == leaf_id]
    n_samples = len(samples_in_leaf)
    avg_salary = samples_in_leaf['salary_thousands'].mean()
    leaf_prediction = samples_in_leaf['prediction'].iloc[0]

    print(f"\n LEAF {leaf_id}:")
    print(f"  Number of samples: {n_samples} people")
    print(f"  Sample IDs: {samples_in_leaf.index.tolist()}")
    print(f"  Real average salary: ${avg_salary:.2f}k")
    print(f"  Tree prediction: ${leaf_prediction:.2f}k")
    print(f"\n  Characteristics of these people:")
    print(f"    Experience: {samples_in_leaf['experience'].min()}-{samples_in_leaf['experience'].max()} years")
    print(f"    Education: {samples_in_leaf['education_years'].min()}-{samples_in_leaf['education_years'].max()} years")
    print(f"    Salaries: ${samples_in_leaf['salary_thousands'].min()}-${samples_in_leaf['salary_thousands'].max()}k")

# ============================================================================
print("\n" + "="*80)
print("STEP 7: TREE VISUALIZATION")
print("="*80)

from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(
    tree_model,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title('Decision Tree - See Samples in Each Node', fontsize=16, fontweight='bold')
plt.show()

print("\n📖 HOW TO READ THE TREE:")
print("""
Each box shows:
  1. Question (if node) or none (if leaf)
  2. squared_error: Squared error at that node
  3. samples: NUMBER OF SAMPLES at that node
  4. value: AVERAGE of salaries of those samples

Example reading:
  If a box says "samples = 5" and "value = 85.0"
  → Means: 5 people reached here
  → The average salary of those 5 people is $85k
  → If it's a LEAF, then $85k is the prediction
""")
