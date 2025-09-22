import numpy as np

def run(seed=0, steps=10):
    rng = np.random.default_rng(seed)
    x = 0.0
    traj = []
    for t in range(steps):
        x += rng.normal(0, 1)
        traj.append((t, x))
    return np.array(traj)

if __name__ == "__main__":
    print(run()[:5])
import numpy as np

def run(seed=0, steps=10):
    rng = np.random.default_rng(seed)
    x = 0.0
    traj = []
    for t in range(steps):
        x += rng.normal(0, 1)
        traj.append((t, x))
    return np.array(traj)

if __name__ == "__main__":
    print(run()[:5])
