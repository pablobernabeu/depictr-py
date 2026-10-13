"""Draw the two images at the top of README.md.

Run from the root of the repository, with the package importable:

    python dev/readme_images.py

The images are served from GitHub raw and kept out of the source distribution,
so rerun this script and commit the result whenever a change alters either
figure. The sizes and resolution match the images the README has always shown.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import depictr as dp  # noqa: E402

OUT = Path("images")


def main() -> None:
    if not OUT.is_dir():
        raise SystemExit("Run this script from the root of depictr-py.")

    ld = dp.lexical_decision()
    distribution = dp.explore_distribution(
        ld, "RT", group="condition", kind="density", legend_inside=True,
        title="Response times by priming condition",
    )
    distribution.save(OUT / "readme-distribution.png", width=8, height=4.6,
                      dpi=150, verbose=False)

    ct = dp.clinical_trial()
    survival = dp.survival_plot(
        ct["time"], ct["event"], group=ct["arm"], risk_table=True,
        legend_inside=True, logrank=True, title="Survival by treatment arm",
    )
    survival.save(OUT / "readme-survival.png", width=8, height=5, dpi=150,
                  verbose=False)


if __name__ == "__main__":
    main()
