from utilities import *
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

Data=ReadVTUFile("/mnt/c/Users/owais/Research_Local/Simvascular_physicsnemo_cardioPINNs/Data1_Stanford4DFlowMRI_25Frames/Velocity_000.vtu")

Point=SinglePointVTK(-45.613,59.689,36.115)
interpolator = vtk.vtkProbeFilter()
interpolator.SetInputData(Point)
interpolator.SetSourceData(Data)
interpolator.Update()
print (interpolator.GetOutput().GetPointData().GetArray("Velocity").GetValue(0))

