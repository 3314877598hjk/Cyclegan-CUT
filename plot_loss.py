import re
import matplotlib.pyplot as plt

log_file = "checkpoints/map2vector_cyclegan/loss_log.txt"

losses = {
    "G_A": [],
    "G_B": [],
    "D_A": [],
    "D_B": [],
    "cycle_A": [],
    "cycle_B": []
}

with open(log_file, "r") as f:
    for line in f:
        for k in losses:
            m = re.search(rf"{k}: ([0-9.]+)", line)
            if m:
                losses[k].append(float(m.group(1)))

plt.figure()
plt.plot(losses["G_A"], label="G_A")
plt.plot(losses["G_B"], label="G_B")
plt.plot(losses["D_A"], label="D_A")
plt.plot(losses["D_B"], label="D_B")

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("CycleGAN Training Loss")
plt.legend()
plt.grid(True)
plt.savefig("cycleGAN_loss.png", dpi=300, bbox_inches="tight")


