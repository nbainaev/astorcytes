"""Grid transitions, fixed goals and shortest-path evaluation distances."""

from collections import deque
import numpy as np


def geometry(spec):
    free = sorted(map(tuple, spec["free"]))
    index = {p: i for i, p in enumerate(free)}
    goal = max(free)
    distance = {goal: 0}
    queue = deque([goal])
    while queue:
        p = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (p[0] + dr, p[1] + dc)
            if q in index and q not in distance:
                distance[q] = distance[p] + 1
                queue.append(q)
    assert len(distance) == len(free)
    transitions = np.empty((len(free), 4), int)
    for p, i in index.items():
        for a, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            transitions[i, a] = index.get((p[0] + dr, p[1] + dc), i)
    return free, index, index[goal], distance, transitions
