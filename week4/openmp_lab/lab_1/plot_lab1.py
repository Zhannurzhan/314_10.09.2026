import pandas as pd, matplotlib.pyplot as plt
d = pd.read_csv("data_saturation.csv")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(d.P, d.mean_seconds, "o-"); ax[0].set(xlabel="Threads P", ylabel="Time (s)", title="Total time")
ax[1].plot(d.P, d.throughput_Msqrt_per_s, "o-"); ax[1].set(xlabel="Threads P", ylabel="M sqrt/s", title="Throughput")
for a in ax: a.axvline(12, ls="--", c="gray"); a.grid(True)   # 12 logical threads
plt.tight_layout(); plt.savefig("lab1_saturation.png", dpi=200)