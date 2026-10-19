"""Pin-checked, deliberately scoped integration with XiaomiMiMo/verl.

Importing this package does not import verl, Ray, or Torch. See docs/verl.md
for the supported contract and the separate GPU validation requirement.
"""

UPSTREAM_URL = "https://github.com/XiaomiMiMo/verl.git"
UPSTREAM_COMMIT = "79090e58195e80a9cfe251c95ae7a78a2132b6e2"
LOSS_NAME = "fusionrl_execution"
ADVANTAGE_NAME = "fusionrl_binary"

