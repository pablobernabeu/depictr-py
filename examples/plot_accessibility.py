"""
Accessibility: a validated colourblind-safe palette
===================================================

depictr defaults to the Okabe-Ito palette and checks that choice rather than
asserting it: a Machado-2009 simulator and a CIE-Lab distance test confirm the
colours stay distinguishable under each colour-vision deficiency.
"""

import depictr as dp

# %%
# The safety report gives the worst-case minimum distance between any two
# palette colours, across normal vision and each deficiency at full severity.
report = dp.palette_safety()
print(report)

# %%
# A grouped density in the default palette.
ld = dp.lexical_decision()
p = dp.explore_distribution(ld, "RT", group="condition", kind="both")
p
