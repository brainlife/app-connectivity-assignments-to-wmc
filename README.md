<h1 align="center">Convert structural connectivity assignments to a white matter classification</h1>

<p align="center">
  A white matter classification (wmc) with one tract per connected region pair, from the streamline assignments of a structural connectome.
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.766"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.766-blue.svg" alt="Run on brainlife.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="MIT license"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Converts structural connectivity streamline assignments (streamline-to-node-pair labels) into a white matter classification (wmc) with one tract per connected region pair, using DIPY and pandas.**

Reads the per-streamline node-pair assignments and the parcellation labels, names each pair as track_<region>_to_<region>, builds a wmc classification and tract files from the tractogram, and generates surface meshes of the parcellation regions for visualisation. Use it to view or analyse the streamlines behind individual connectome edges; the tractogram and parcellation must be the ones used to compute the assignments.

| | |
|---|---|
| **Inputs** | `neuro/labels` · `neuro/track/tck` · `neuro/dwi` · `neuro/parcellation/volume` |
| **Outputs** | `neuro/wmc` |
| **Container** | `brainlife/pyafq:0.4.1` |
| **Resources** | 8 CPUs · 1 h walltime |
| **Run it** | [doi.org/10.25663/brainlife.app.766](https://doi.org/10.25663/brainlife.app.766) |

#### Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `assignments` | `neuro/labels` (connectivity_assignments) | yes | Streamline-to-node-pair assignments from a structural connectivity App (index and pair names). |
| `track` | `neuro/track/tck` | yes | Tractogram the assignments refer to. |
| `dwi` | `neuro/dwi` | yes | DWI used as the reference space when loading the tractogram. |
| `parcellation` | `neuro/parcellation/volume` | yes | Parcellation (with label.json) whose regions define the connectome nodes. |

#### Outputs

| Output | Datatype | Description |
|---|---|---|
| `wmc` | `neuro/wmc` | White matter classification with one tract per connected region pair, plus tract and region surface files. |

<!-- brainlife-app:end -->

## Running the App

A structural connectivity App assigns each streamline to the pair of parcellation regions it connects. This App reads those assignments and the parcellation labels, names each pair `track_<region>_to_<region>` (region order normalised so that A→B and B→A are one tract), drops unassigned (`not-classified`) streamlines, and builds a wmc classification and per-tract files from the tractogram. It also meshes every parcellation region so the tracts can be viewed with their endpoints in the brainlife.io visualiser.

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.766](https://doi.org/10.25663/brainlife.app.766) and choose, from the same connectivity run:

- the assignments (`neuro/labels` tagged `connectivity_assignments`),
- the tractogram they refer to (`neuro/track/tck`),
- the DWI used as reference space (`neuro/dwi`),
- the parcellation whose regions are the connectome nodes (`neuro/parcellation/volume`, with `label.json`).

Press **Execute**. The result is archived to your project as a `neuro/wmc` dataset that any App taking a wmc input (tract statistics, conversions, visualisation) can use.

### Locally

You need [Singularity/Apptainer](https://apptainer.org/). The App runs `wmc-generation.py` in `brainlife/pyafq:0.4.1` (DIPY, nibabel, pandas, SciPy) and `parcellation2vtks.py` in `brainlife/pythonvtk:1.1` (VTK).

1. Clone this repository and check out the registered version: `git checkout pyafq-0.4.1_v1.0`.
2. Create `config.json` with the paths to your inputs:

   ```json
   {
       "index": "/input/assignments/index.csv",
       "name": "/input/assignments/names.csv",
       "track": "/input/track/track.tck",
       "dwi": "/input/dwi/dwi.nii.gz",
       "parcellation": "/input/parcellation/parc.nii.gz",
       "labels": "/input/parcellation/label.json"
   }
   ```

   `index` and `name` are the two single-column files of the assignments dataset (streamline index and region pair such as `12_45`); `labels` is the parcellation's `label.json` with `voxel_value` and `name` for each region.

3. Run the App:

   ```bash
   ./main
   ```

## Outputs

```text
wmc/
├── classification.mat   names (one per tract) and per-streamline tract index (0 = unassigned)
├── tracts/
│   ├── tracts.json      tract list with names, colours and file names
│   └── <index>.json     streamline coordinates of each tract
└── surfaces/
    ├── index.json       region list
    └── <value>.<name>.vtk   one mesh per parcellation region
```

Tract and region colours are assigned at random when the parcellation labels have none.

## Troubleshooting

- **`failed`** (no `wmc/classification.mat`): the assignments, tractogram and parcellation must come from the same connectivity run; a pair label missing from `label.json` or a streamline count that does not match the tractogram stops the conversion.
- **Surface generation is slow**: listing the parcellation's unique values with VTK is slow on large or high-resolution parcellations; this step is skipped on re-runs once `wmc/surfaces/index.json` exists.

## Citing

> [!TIP]
> If you use this App in your work, please cite the App, the software it builds on and brainlife.io (below). GitHub's **Cite this repository** button gives the App citation in APA and BibTeX.

- Convert assignments from structural connectivity to wmc datatype. brainlife.io App. [doi:10.25663/brainlife.app.766](https://doi.org/10.25663/brainlife.app.766)
- Garyfallidis et al. Dipy, a library for the analysis of diffusion MRI data. *Frontiers in Neuroinformatics* 8 (2014). [doi:10.3389/fninf.2014.00008](https://doi.org/10.3389/fninf.2014.00008)
- Kruper et al. Evaluating the Reliability of Human Brain White Matter Tractometry. *Aperture Neuro* 2021 (2021). [doi:10.52294/e6198273-b8e3-4b63-babb-6e6b0da10669](https://doi.org/10.52294/e6198273-b8e3-4b63-babb-6e6b0da10669)
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Scientific Data* 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Authors

- Brad Caron

## Funding

[![Wellcome-226486/Z/22/Z](https://img.shields.io/badge/Wellcome-226486%2FZ%2F22%2FZ-purple.svg)](https://wellcome.org/)
[![NIH-NINDS-U24NS140384](https://img.shields.io/badge/NIH_NINDS-U24NS140384-green.svg)](https://reporter.nih.gov/project-details/11261037)
[![NIH-NINDS-UM1NS132207](https://img.shields.io/badge/NIH_NINDS-UM1NS132207-green.svg)](https://reporter.nih.gov/project-details/11411328)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://reporter.nih.gov/project-details/10262925)
[![NIH-NIBIB-R01EB030896](https://img.shields.io/badge/NIH_NIBIB-R01EB030896-green.svg)](https://reporter.nih.gov/project-details/10428625)
[![NIH-NIMH-R01MH126699](https://img.shields.io/badge/NIH_NIMH-R01MH126699-green.svg)](https://reporter.nih.gov/project-details/10460628)
[![NIH-NIDCD-R01DC020088](https://img.shields.io/badge/NIH_NIDCD-R01DC020088-green.svg)](https://reporter.nih.gov/project-details/11140284)
[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-2203524](https://img.shields.io/badge/NSF_BCS-2203524-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2203524)
[![NSF-IIS-1636893](https://img.shields.io/badge/NSF_IIS-1636893-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-OAC-2148729](https://img.shields.io/badge/NSF_OAC-2148729-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148729)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NSF-IIS-2148700](https://img.shields.io/badge/NSF_IIS-2148700-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148700)
[![NSF-OAC-1916518](https://img.shields.io/badge/NSF_OAC-1916518-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1916518)

<sub>brainlife.io is publicly funded, and acknowledging that support helps keep it running. This App was developed with support from Wellcome, the U.S. National Institutes of Health (including the BRAIN Initiative BRAIN CONNECTS program: The Axonal Projectome EXchange, U24NS140384, and the Center for Mesoscale Connectomics, UM1NS132207) and the U.S. National Science Foundation.</sub>

## License

[MIT](LICENSE).
