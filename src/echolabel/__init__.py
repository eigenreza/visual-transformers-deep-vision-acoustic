"""Species detection in in-trawl camera imagery and the data pipeline around it.

The package builds the annotation dataset, trains, evaluates and calibrates
object detectors, applies the selected detector to whole camera transects and
forms time series of counts and camera depth. It also holds the infrastructure
for integrating the detections with a multi-frequency echosounder record.
"""

__version__ = "0.1.0"
