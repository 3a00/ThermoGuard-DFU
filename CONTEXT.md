# ThermoGuard-DFU

Automated diabetic foot ulcer (DFU) risk assessment from plantar thermal imaging using deep learning.

## Language

**Plantar Thermogram**:
A 2D spatial matrix capturing absolute skin surface temperatures across the sole of a human foot.
_Avoid_: Thermal photo, heat map, colormap image, RGB thermogram

**Angiosome**:
An anatomical vascular territory of the foot sole perfused by a specific source artery (LCA, LPA, MCA, or MPA).
_Avoid_: Foot zone, quadrant, region of interest, heat sector

**Thermal Change Index (TCI)**:
A quantitative physiological metric computed from temperature discrepancies between corresponding angiosomes of contralateral feet.
_Avoid_: Temperature score, asymmetry index, ulcer score

**Severity Grade**:
A 3-level clinical risk categorization (`Healthy`, `Low_Severity`, `High_Severity`) derived by grouping TCI severity grades into clinically actionable stages.
_Avoid_: 6-class grade, Wagner grade, ulcer stage

**Contralateral Foot Pair**:
The paired left and right feet of an individual subject evaluated together to detect pathological thermal asymmetry.
_Avoid_: Both feet, bilateral feet, foot set
