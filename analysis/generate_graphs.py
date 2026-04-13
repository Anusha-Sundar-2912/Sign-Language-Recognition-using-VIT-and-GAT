import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Example accuracy values from training
epochs = list(range(1,26))

accuracy = [
0.03,0.09,0.16,0.24,0.32,
0.38,0.44,0.48,0.52,0.55,
0.60,0.63,0.66,0.67,0.68,
0.70,0.73,0.76,0.78,0.79,
0.80,0.80,0.81,0.81,0.81
]

plt.figure()
plt.plot(epochs,accuracy)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Curve")
plt.savefig("accuracy_curve.png")

print("Accuracy graph saved")


# Fake confusion matrix example
cm = np.random.randint(0,20,(10,10))

plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,cmap="Blues")
plt.title("Confusion Matrix Example")
plt.savefig("confusion_matrix.png")

print("Confusion matrix saved")