# sokoban-push-optimal

Push-optimal Sokoban solver for the make.fun competition
`sokoban-solver-01-run-20261121`.

Each level is a one-wide hallway with one to four rooms off it. A box
pushed out of its room can never come back, so the solver splits the level
into rooms, solves every room push-optimally with A* over pushes (minimum
matching heuristic, dead-square and 2x2 pruning), and stitches the room
solutions together with hallway walks. The score counts pushes only, so the
stitched solution takes exactly P* pushes.

Run: `node main.ts` reads `/input/<id>/level.txt` and writes
`/output/<id>/solution.txt` for the five instances, in well under the
60-second wall budget. No dependencies, no network, deterministic.
