# Recommended Web Tool Workflow

This workflow is based on the current web-tool implementation of regular region growing, interface-constrained region growing (ICRG), and distance-weighted dense label propagation.

The web tool now lives in the standalone `rock_seg_3d_web` package. Launch and frontend-development details are in [`rock_seg_3d_web/README.md`](../rock_seg_3d_web/README.md).

## 1. Start a project early

Upload or import the point cloud, then use **Save As** to create a `.rd3dproj` project file.
After a save target has been chosen, **Save Project** updates the same project file when the browser supports file-system write access.

The `.rd3dproj` file is a ZIP archive with a readable `project.json` manifest plus binary artifacts. It is intended to preserve both the analysis settings and the recoverable workflow state: raw/working point clouds, seeds, interface metadata, region-growing outputs, label-propagated segmentation, available mesh products, and analysis results.

Save the project at major milestones:

- after seed selection
- after manual interface creation
- after successful ICRG
- after final label propagation
- after mesh reconstruction or analysis

## 2. Set seeds first

Add at least one rock seed and one pedestal/support seed.
Auto seeds can be useful as a starting point, but manually inspect and correct them before running segmentation.

Good seeds are more important than aggressive parameter tuning. Poor seeds can cause branch growth errors that later label propagation may spread into the dense cloud.

## 3. Run regular region growing first

Use **Run Region Growing** before ICRG.
Inspect **RG Result** rather than jumping directly to **Segmented**.

The RG Result view shows the voxel-stage result before dense label propagation. Use the branch colors and seed branch counts to check whether each seed grew into the expected part of the point cloud.

## 4. Run regular region growing as a fast diagnostic step

The purpose of regular region growing is not necessarily to produce the final segmentation. It mainly helps with two things:

1. generate an automatic interface that can be used as a starting point for manual refinement
2. help the user understand how the current seeds and region-growing parameters behave

For this step, keep **Voxel** relatively large enough to run quickly. A coarse voxel size is usually acceptable because the automatic interface is only a draft and will be refined manually.

Recommended diagnostic starting values:

| Parameter | Starting value | Purpose |
| --- | ---: | --- |
| Voxel | `0.02-0.05` | Coarser downsampling for faster diagnostic region growing |
| Neighborhood radius | `0.04-0.10` | Neighbor-search radius, usually at least `2x` voxel size |
| Normal neighbors | `50` | Maximum neighbors for normal estimation used by smoothness and curvature |
| Smoothness | `0.9` | Normal-alignment threshold for accepting neighboring points |
| Curvature | `0.1` | Local curvature threshold for accepting neighboring points |

Tune parameters in this order:

1. Adjust **Smoothness** and **Curvature** first. These control whether growth accepts points based on local surface continuity.
2. Adjust **Neighborhood radius** second. This controls whether the algorithm can physically reach nearby points.
3. Leave **Normal neighbors** near the default unless normals appear unstable or the cloud is very sparse or noisy.

If labels cross the interface:

- increase **Smoothness**
- decrease **Curvature**
- decrease **Neighborhood radius**

If rock growth stops too early:

- lower **Smoothness** slightly
- increase **Curvature** slightly
- increase **Neighborhood radius** carefully

## 5. Create and refine the manual interface

After regular region growing, use the automatic interface as a draft if it looks close.
For difficult contacts, prefer:

1. **Start From Auto**
2. refine the interface manually
3. **Save as Manual**

The saved manual interface is the constraint used by **Run ICRG**.

## 6. Run ICRG

Use **Run ICRG** after saving the manual interface.

ICRG is the preferred segmentation step near the rock/support boundary because **Interface exclusion radius** prevents region growth from crossing the manual interface.

This is the step where parameters should be fine tuned for the final segmentation.

Compared with the fast diagnostic region-growing step:

- reduce **Voxel** to preserve more contact detail
- reduce **Neighborhood radius** at the same time, so growth does not jump across narrow gaps or contacts
- reduce **Interface exclusion radius** so the final boundary is not overly conservative

Useful ICRG parameter heuristics:

| Relationship | Recommendation |
| --- | --- |
| **Neighborhood radius** vs. **Voxel** | Keep neighborhood radius at least `2x` voxel size. |
| **Interface exclusion radius** vs. **Voxel** | Keep interface exclusion radius larger than voxel size. |
| **Label propagation** vs. **Neighborhood radius** | Keep label propagation radius close to or smaller than the final neighborhood radius. |

If the region hardly grows:

- increase **Neighborhood radius** first
- then slightly lower **Smoothness** if growth is still too strict
- then slightly increase **Curvature** if rough rock surfaces are still being missed

If ICRG still crosses the contact:

- increase **Smoothness**
- decrease **Curvature**
- decrease **Neighborhood radius**
- increase **Interface exclusion radius**

If too much valid rock near the contact stays unlabeled:

- decrease **Interface exclusion radius**, while keeping it larger than voxel size
- increase **Neighborhood radius** if the region cannot reach valid neighboring rock points

These relationships are practical starting rules. Very sparse or uneven point clouds may need a larger neighborhood radius, but increasing it should be done carefully because it can also increase overspill risk.

## 7. Run label propagation last

Use **Run Label Propagation** only after the voxel-stage RG or ICRG result looks correct.

Label propagation performs distance-weighted dense completion. It fills or transfers labels onto the dense point cloud using nearby labeled points.
Keep the **Label propagation** radius conservative, usually close to the region-growing neighborhood radius or smaller.

If the label-propagation radius is too large, it can undo a good ICRG boundary by filling labels across the interface.

## 8. Inspect the segmented result

Use **Segmented** after label propagation.

- Use two-color mode for the final rock/support classification.
- Use multi-color mode when debugging which seed branch controlled each area.

If the final dense labels look wrong, go back to **RG Result** first. Fix the voxel-stage region growing or ICRG result before tuning label propagation.

## 9. Prepare meshes after segmentation is stable

Once segmentation is stable, run mesh preparation, normals, reconstruction, and analysis.

Do not tune mesh settings before the segmentation boundary is correct. Mesh and analysis quality depend strongly on the segmentation result.

The web tool now keeps rock and pedestal preparation states separately. In **4. Mesh Preparation**, choose the target before running mesh-prep tools:

- **Prepare Rock Mesh** uses rock object points plus the interpolated interface-fill points needed for rock reconstruction.
- **Prepare Pedestal Mesh** uses only pedestal/support object points. Interface path points are not included in the pedestal preparation state.

Use **Run Preparation** to open an existing prepared target when it already exists. Use **Reset** when you want to reload from the label-propagated segmentation. For pedestal reset, select which pedestal seed branches to include, or choose all label-propagated pedestal points.

For pedestal cleanup, use these tools before reconstruction:

- **Height Above Ground** selects likely vegetation points above local ground.
- **Roughness** computes local best-fit-plane roughness, displays a heatmap, then selects points above the chosen threshold.
- **Manual Removal** removes all prepared points projected inside the polygon, including points hidden behind the front surface.

For rock cleanup, **Manual Removal** can remove rock object points, interpolated interface-fill points, and visible interface path points from the mesh-prep state.

In **5. Normals**, choose **Rock Normals** or **Pedestal Normals** to match the current mesh-prep target.

## 10. Reconstruct rock and pedestal surfaces

In **6. Reconstruction**, choose the target before reconstruction:

- **Rock** uses the existing Poisson reconstruction workflow.
- **Pedestal** uses the local-plane filled-hole surface reconstruction workflow.

Use **Load Rock + Pedestal** to inspect both targets together. The combined view uses each target's reconstructed mesh when available. If a mesh is missing, it falls back to that target's segmented point cloud. The button is disabled unless both rock and pedestal have at least one available source: either a reconstructed mesh or segmented point-cloud points.

Analysis remains rock-only in the current web tool. Run **Analyze** after rock reconstruction when the rock mesh is ready.

## 11. Use measurement as a temporary inspection tool

The **Measurement** panel is useful for checking distances while reviewing segmentation or prepared point clouds.
It uses the currently displayed point-cloud coordinates and is not saved into `.rd3dproj` projects.

Use **Shift + Left Click** to select two points and **Shift + Right Click** to remove a selected measurement point.

## Practical recommendation

Use regular region growing as a diagnostic step, use manual-interface ICRG as the main segmentation step, and use label propagation only after the voxel-stage result looks correct.

In short:

```text
Upload or import
-> choose seeds
-> Run Region Growing
-> inspect RG Result
-> create or refine manual interface
-> Run ICRG
-> inspect RG Result again
-> Run Label Propagation
-> inspect Segmented
-> prepare mesh
-> reconstruct rock and/or pedestal
-> load rock + pedestal together when needed
-> analyze rock geometry
-> Save Project
```
