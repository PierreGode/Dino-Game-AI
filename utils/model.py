import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils.helperfuncs import get_matrix_from_img

pos_examples_path = "/home/pierre/Dino-Game-AI/data/pos examples"
neg_examples_path = "/home/pierre/Dino-Game-AI/data/neg examples"

# Extracting features from images
pos_features = get_matrix_from_img(pos_examples_path)
neg_features = get_matrix_from_img(neg_examples_path)

# Creating target labels for positive and negative examples
pos_target = np.ones(pos_features.shape[0], dtype="int64")
neg_target = np.zeros(neg_features.shape[0], dtype="int64")

# Combining features and labels
X = np.concatenate([pos_features, neg_features], axis=0)
y = np.concatenate([pos_target, neg_target], axis=0)

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train the logistic regression model with increased max_iter
model = LogisticRegression(solver="lbfgs", max_iter=1000)
model.fit(X_scaled, y)
