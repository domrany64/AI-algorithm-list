class Category:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children if children is not None else []

    def add_children(self, children):
        self.children.extend(children)

# Function to generate ASCII tree recursively
def generate_ascii_tree(category, prefix='', is_last=True):
    print(prefix + ("└── " if is_last else "├── ") + category.name)
    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(category.children):
        generate_ascii_tree(child, child_prefix, i == len(category.children) - 1)

# Create the category hierarchy
ai_category = Category("Artificial Intelligence (AI)", [
    Category("Machine Learning", [
        Category("Deep Learning", [
            Category("Artificial Neural Networks", [
                Category("Convolutional Neural Networks (CNN)"),
                Category("Recurrent Neural Networks (RNN)"),
                Category("Long Short-Term Memory (LSTM)"),
                Category("Gated Recurrent Unit (GRU)"),
                Category("Autoencoders"),
                Category("Generative Adversarial Networks (GAN)"),
                Category("Transformer Networks")
            ])
        ]),
        Category("Supervised Learning", [
            Category("Support Vector Machines (SVM)"),
            Category("Decision Trees"),
            Category("Random Forests"),
            Category("K-Nearest Neighbors (KNN)"),
            Category("Naive Bayes"),
            Category("Regression Techniques", [
                Category("Linear Regression"),
                Category("Logistic Regression")
            ]),
            Category("Ensemble Methods", [
                Category("Bagging"),
                Category("Boosting", [
                    Category("Gradient Boosting Machines (GBM)"),
                    Category("AdaBoost")
                ]),
                Category("Stacking")
            ])
        ]),
        Category("Unsupervised Learning", [
            Category("Clustering Algorithms", [
                Category("K-Means"),
                Category("DBSCAN clustering algorithm"),
                Category("Gaussian Mixture Model algorithm"),
                Category("BIRCH algorithm"),
                Category("Affinity Propagation clustering algorithm"),
                Category("Mean-Shift clustering algorithm"),
                Category("OPTICS algorithm"),
                Category("Agglomerative Hierarchy clustering algorithm")
            ]),
            Category("Dimensionality Reduction Techniques", [
                Category("Principal Component Analysis (PCA)"),
                Category("Linear Discriminant Analysis (LDA)"),
                Category("t-Distributed Stochastic Neighbor Embedding (t-SNE)"),
                Category("Autoencoders")
            ])
        ]),
        Category("Reinforcement Learning", [
            Category("Deep Reinforcement Learning"),
            Category("Monte Carlo Tree Search (MCTS)"),
            Category("Multi-agent Reinforcement Learning")
        ]),
        Category("Hyperparameter Optimization", [
            Category("Grid Search"),
            Category("Random Search"),
            Category("Bayesian Optimization")
        ]),
        Category("Model Interpretability", [
            Category("SHAP Values"),
            Category("LIME"),
            Category("Feature Importance Analysis")
        ]),
        Category("Other ML Concepts", [
            Category("Data Preprocessing Techniques"),
            Category("Evaluation Metrics"),
            Category("Time Series Analysis", [
                Category("ARIMA"),
                Category("Exponential Smoothing"),
                Category("Prophet")
            ]),
            Category("Anomaly Detection", [
                Category("Isolation Forest"),
                Category("One-Class SVM"),
                Category("Local Outlier Factor")
            ]),
            Category("Sequential Data Processing", [
                Category("Hidden Markov Models"),
                Category("Recurrent Neural Networks"),
                Category("Long Short-Term Memory")
            ]),
            Category("Graph-based Learning", [
                Category("Graph Convolutional Networks"),
                Category("Graph Attention Networks"),
                Category("Graph Embedding")
            ]),
            Category("Causal Inference", [
                Category("Do-Calculus"),
                Category("Propensity Score Matching"),
                Category("Structural Equation Modeling")
            ])
        ])
    ]),
    Category("Natural Language Processing", [
        Category("Syntax Analysis", [
            Category("Parsing"),
            Category("Part-of-speech tagging (POS)")
        ]),
        Category("Sentiment Analysis"),
        Category("Machine Translation"),
        Category("Speech Recognition"),
        Category("Transformer-based Models", [
            Category("BERT"),
            Category("GPT"),
            Category("T5")
        ])
    ]),
    Category("Computer Vision", [
        Category("Feature Extraction", [
            Category("SIFT"),
            Category("HOG descriptors")
        ]),
        Category("Video Analysis"),
        Category("Scene Reconstruction"),
        Category("Augmented Reality")
    ]),
    Category("Expert Systems", [
        Category("Explanation and Justification Facilities"),
        Category("User Interface"),
        Category("Inference Engine"),
        Category("Knowledge Base"),
        Category("Rule Engine")
    ]),
    Category("Robotics", [
        Category("Autonomous Navigation"),
        Category("Human-Robot Interaction"),
        Category("Motion Planning"),
        Category("Manipulation"),
        Category("Perception")
    ]),
   Category("Bayesian Networks (BN)"),
    Category("Fuzzy Logic (FL)"),
    Category("Genetic Algorithms (GA)"),
    Category("Knowledge Representation and Reasoning", [
        Category("Semantic Networks"),
        Category("Frames"),
        Category("Ontologies")
    ]),
    Category("Deep Q-Networks (DQN)"),
    Category("Variational Inference (VI)"),
    Category("Graph Neural Networks (GNNs)"),
    Category("Meta-Learning Algorithms"),
    Category("Evolutionary Strategies (ES)"),
    Category("Sparse Coding"),
    Category("Temporal-Difference Learning (TD Learning)")
])

# Print the ASCII tree starting from the top-level category
generate_ascii_tree(ai_category)

