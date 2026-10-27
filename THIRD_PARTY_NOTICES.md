# Third-party notices

MIT applies to original gpuGWR code only. The owner selected MIT on 2026-10-28.

The independent mathematical implementation acknowledges the PySAL mgwr,
spglm and spreg authors. Their preserved BSD-3-Clause reference files are
described in [Reference/ATTRIBUTION.md](Reference/ATTRIBUTION.md), with their
original license texts. They are not bundled in the installed gpuGWR wheel.
The reviewed publication includes only the reference notices and license texts,
not the offline wheel cache or historical source archives.

Georgia data is distributed by libpysal under its preserved BSD-3-Clause
license, [Reference/libpysal_LICENSE.txt](Reference/libpysal_LICENSE.txt).
Columbus, Baltimore and Boston analysis copies derive from spData 2.3.5,
commit `6d9c2a8d087822463f6571016826fe45efa749c0`, declared CC0 by its
original DESCRIPTION. See [dataset notes](Documentation/Contract04/DATASETS.md).
These statements do not relicense the underlying publications.

The Brazil Gist has no established license in the supplied provenance.
Its source geometry and derived input table are excluded from the publication
candidate. The notebook requests the original pinned upstream URL and checks
the exact supplied-file SHA-256; remote byte identity remains unverified until
that acquisition succeeds. MIT does not grant rights to that third-party data.

NumPy, SciPy, pandas, pyproj, GeoPandas, CuPy and the other dependencies retain
their respective licenses. They are installed separately and are not vendored.
The installation records exact resolved versions for reproducibility.
