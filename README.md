<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

**Converts structural connectivity streamline assignments (streamline-to-node-pair labels) into a white matter classification (wmc) with one tract per connected region pair, using DIPY and pandas.**

Reads the per-streamline node-pair assignments and the parcellation labels, names each pair as track_<region>_to_<region>, builds a wmc classification and tract files from the tractogram, and generates surface meshes of the parcellation regions for visualisation. Use it to view or analyse the streamlines behind individual connectome edges; the tractogram and parcellation must be the ones used to compute the assignments.

Run it on brainlife.io: [https://doi.org/10.25663/brainlife.app.766](https://doi.org/10.25663/brainlife.app.766)

| Input | Datatype | Required | Description |
|---|---|---|---|
| `assignments` | `neuro/labels` (connectivity_assignments) | yes | Streamline-to-node-pair assignments from a structural connectivity App (index and pair names). |
| `track` | `neuro/track/tck` | yes | Tractogram the assignments refer to. |
| `dwi` | `neuro/dwi` | yes | DWI used as the reference space when loading the tractogram. |
| `parcellation` | `neuro/parcellation/volume` | yes | Parcellation (with label.json) whose regions define the connectome nodes. |

| Output | Datatype | Description |
|---|---|---|
| `wmc` | `neuro/wmc` | White matter classification with one tract per connected region pair, plus tract and region surface files. |

Runtime: container `brainlife/pyafq:0.4.1`; 8 CPUs; 1 h walltime.

<!-- brainlife-app:end -->

# app-connectivity-assignments-to-wmc