# Internal_SolarCell_Resistance-1D
Extricating the sheet resistance of transparent conductive oxide from series resistance to obtain actual internal/intrinsic of the active layer.

Simulation in Python. Main packages required: math, numpy, scipy, joblib and matplotlib.

### Parallel processing employed using joblib.

![Joblib](threads_vs_time_temp.png)
-----
The cell is discretized into multiple cells in series.
![Schematic](Schematic.png)

-----
Calculates Root Mean Squared Error for a set of Intrinsic series resistances. The minimum gives the cell's internal resistance.
![RMSE](RMSerror_vs_Rintrinsic.png)

-----
Comparison with Rseries vs Rintrinsic.

![plot](IV_RintrinsicDetermination_v3.png)
