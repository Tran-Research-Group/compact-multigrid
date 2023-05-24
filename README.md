# compact-multigrid
Base Gymnasium environment for multirgrid environments with location-based compact observation representations.

## Environments
Provides several gyn environments:
- `BaseMultigrid`: Base environment for all the other environments in the package.
- `BaseCtf`: Base environment for Capture-the-Flag environments.
- `Ctf1v1`: 1v1 CtF Environment

## Usage
Simply, 
```
from compact_multigrid import BaseMultigrid, BaseCtf, Ctf1v1
```
and you are good to go!