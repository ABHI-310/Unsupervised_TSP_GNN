import json
import matplotlib.pyplot as plt

with open("training_log.json") as f:
    log = json.load(f)

plt.plot(log["loss"], label="Surrogate loss")
plt.plot(log["avg_tour"],label="Avg tour length")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("UTSP Training Curves")
plt.show()