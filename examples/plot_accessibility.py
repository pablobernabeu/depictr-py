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
# A grouped distribution in the default palette -- histogram with its density
# curve. ``legend_inside`` tucks the legend into the empty top-right corner
# instead of a separate margin.
ld = dp.lexical_decision()
p = dp.explore_distribution(ld, "RT", group="condition", kind="both",
                            legend_inside=True)
p
