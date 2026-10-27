# Preserved third-party reference material

The files under mgwr/, spglm/, and spreg/ are original PySAL source files,
preserved unchanged for inspection and execution. They are not part of the
independent gpugwr mathematical implementation and are excluded from its
formatting/readability audit so that upstream bytes are not changed.

| Project | Tag/version | Exact retrieved tree/commit |
|---|---|---|
| [mgwr](https://github.com/pysal/mgwr) | v2.2.1 | 423726a3a949147fd5416bef0fec33021a51cbf3 |
| [spglm](https://github.com/pysal/spglm) | v1.1.0 | ee2d424156118278a3c3f292bf55a9f5d3ff6f7e |
| [spreg](https://github.com/pysal/spreg) | v1.8.3 | dc529360ec6ce02c6ef9afe0c15162ab8b6ccb26 |

Copyright belongs to the respective PySAL contributors. Preserve
[mgwr/LICENSE](mgwr/LICENSE), [spglm/LICENSE.txt](spglm/LICENSE.txt),
and [spreg/LICENSE.txt](spreg/LICENSE.txt). Each uses the BSD 3-Clause terms.
The independent implementation follows the mathematical conventions of these
sources and acknowledges their authors, particularly the mgwr and spglm
implementations of GWR, diagnostics, local collinearity, and search behavior.

[source_manifest.json](source_manifest.json) records each preserved source's
Git blob hash and source repository/ref. blobs_*.json preserve API-returned
base64 bytes; [unpack_reference.py](../Scripts/unpack_reference.py) decodes
them and checks Git object hashes before writing. The 76 preserved files
include the runtime modules, package metadata, licenses, and Georgia notebook;
they are a scoped source snapshot, not a full repository clone. They are
imported directly only in the dedicated reference module. Their original
build files are preserved but do not need to be executed.

The libpysal data manifest records the official Georgia directory and the
original local files are hash-checked against it. Its license is retained in
[libpysal_LICENSE.txt](libpysal_LICENSE.txt).

Original cached dependency wheels under wheels/ retain their own metadata and
license notices. [wheel_manifest.json](wheel_manifest.json) records local cache
provenance and SHA-256 hashes. These are a Windows/Python 3.12 offline execution
bundle, not a public package release. Source code is not edited to force any
numerical result.
