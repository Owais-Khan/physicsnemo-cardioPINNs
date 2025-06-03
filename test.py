from utilities import *

surface=ReadSTLFile("./stl_files/Data1_Stanford4DFlowMRI-mesh-complete/mesh-combined.stl")
centroid=GetCentroid(surface)
centroid=(-1*centroid[0],-1*centroid[1],-1*centroid[2])
translation=vtk.vtkTransform()
translation.Translate(centroid)


transform_surface=vtk.vtkTransformPolyDataFilter()
transform_surface.SetInputData(surface)
transform_surface.SetTransform(translation)
transform_surface.Update()
transform_surface=transform_surface.GetOutput()

scaling=vtk.vtkTransform()
scaling.Scale((0.1,0.1,0.1))

scale_surface=vtk.vtkTransformPolyDataFilter()
scale_surface.SetInputData(transform_surface)
scale_surface.SetTransform(scaling)
scale_surface.Update()
scale_surface=scale_surface.GetOutput()


WriteVTPFile("test.vtp",scale_surface)

exit(1)
print ((surface.GetPoints().GetPoint(0)))
print(dir(surface.SetPoints(0)))
exit(1)
print (dir(surface))
exit(1)

print (surface.GetBounds())
exit(1)

#surfaceNormal_=SurfaceNormals(surface)
#WriteVTPFile("test.vtp",surfaceNormal_)
