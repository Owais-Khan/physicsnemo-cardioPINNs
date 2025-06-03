from utilities import *

surface=ReadSTLFile("./stl_files/Data1_Stanford4DFlowMRI-mesh-complete/mesh-combined.stl")
print ((surface.GetPoints().GetPoint(0)))
print(dir(surface.SetPoints(0)))
exit(1)
print (dir(surface))
exit(1)

print (surface.GetBounds())
exit(1)

#surfaceNormal_=SurfaceNormals(surface)
#WriteVTPFile("test.vtp",surfaceNormal_)
