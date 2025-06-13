import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
from glob import glob
from scipy.spatial import distance as DISTANCE
from scipy.stats import iqr as IQR
from scipy.stats import kurtosis as KURTOSIS
from scipy.stats import skew as SKEWNESS
from scipy.stats import mode as MODE
from sympy import Symbol, sqrt, Max
############ PhysicsNeMO Functions #########

# inlet velocity profile
def circular_parabola(x, y, z, center, normal, radius, max_vel):
    centered_x = x - center[0]
    centered_y = y - center[1]
    centered_z = z - center[2]
    distance = sqrt(centered_x**2 + centered_y**2 + centered_z**2)
    parabola = max_vel * Max((1 - (distance / radius) ** 2), 0)
    return normal[0] * parabola, normal[1] * parabola, normal[2] * parabola

    # normalize meshes
def normalize_mesh(mesh, center, scale):
    mesh = mesh.translate([-c for c in center])
    mesh = mesh.scale(scale)
    return mesh

def normalize_mesh_vtk(mesh,center,scale):
    for i in range(mesh.GetNumberOfPoints()):
        point_=mesh.GetPoints().GetPoint(i)
        pointnewX_=(point_[0]-center[0])*scale
        pointnewY_=(point_[1]-center[1])*scale
        pointnewZ_=(point_[2]-center[2])*scale
        pointNew_=(pointnewX_,pointnewY_,pointnewZ_)
        mesh.GetPoints().SetPoint(i,pointNew_)
        mesh.Modified()
    return mesh


def ProjectData(InputMesh=None,SourceMesh=None): #Project data from SourceMesh to InputMesh
    ProjectedSurface=vtk.vtkProbeFilter()
    ProjectedSurface.SetInputData(InputMesh)
    ProjectedSurface.SetSourceData(SourceMesh)
    ProjectedSurface.Update()
    return ProjectedSurface.GetOutput()

# normalize invars
def normalize_invar(invar, center, scale, dims=2):
    invar["x"] -= center[0]
    invar["y"] -= center[1]
    invar["z"] -= center[2]
    invar["x"] *= scale
    invar["y"] *= scale
    invar["z"] *= scale
    if "area" in invar.keys():
        invar["area"] *= scale**dims
    return invar

def TranslateScalePolyData(Surface,TranslationVector=(0,0,0),ScalingVector=(1,1,1)):
   
    #Create a Translation Function
    translation=vtk.vtkTransform()
    translation.Translate(TranslationVector)
    
    #Apply the Translation to the Surface    
    transformed_surface=vtk.vtkTransformPolyDataFilter()
    transformed_surface.SetInputData(Surface)
    transformed_surface.SetTransform(translation)
    transformed_surface.Update()
    transformed_surface=transformed_surface.GetOutput()

    #Scale the surface
    scaled_surface=ScalePolyData(transformed_surface,ScalingVector)
    
    return scaled_surface

def ScalePolyData(Surface,ScalingArray):
    #Create a Translation Function
    scaling=vtk.vtkTransform()
    scaling.Scale(ScalingArray)
    #Apply the Translation to the Surface
    scaled_surface=vtk.vtkTransformPolyDataFilter()
    scaled_surface.SetInputData(Surface)
    scaled_surface.SetTransform(scaling)
    scaled_surface.Update()
    scaled_surface=scaled_surface.GetOutput()
    return scaled_surface


#Get the distance from the surface to the interior points
def GetDistanceFromWall(Volume,Surface=None): 
    if Surface is None: Surface=ExtractSurface(Volume)
   
    #Signed Distance Array
    signedDistance=np.zeros(Volume.GetNumberOfPoints())

    #Create Distance Function
    implicitPolyDataDistance = vtk.vtkImplicitPolyDataDistance()
    implicitPolyDataDistance.SetInput(Surface)

    # Evaluate the signed distance function at all of the grid points
    for pointId in range(Volume.GetNumberOfPoints()):
        signedDistance_ = implicitPolyDataDistance.EvaluateFunction(Volume.GetPoint(pointId))
        signedDistance[pointId]=np.absolute(signedDistance_)

    #Note:  -ve distance ==> inside the volume
    #       +ve distance ==> outside the volume
    #        0 distance  ==> on the surface

    return signedDistance


#Get the data within a certain distance
def GetVelocityAwayFromWall(Volume,DistanceThreshold,ArrayName):
    #Create an array to store velocity
    u=[]; v=[]; w=[]; x=[]; y=[]; z=[]

    Velocity=vtk_to_numpy(Volume.GetPointData().GetArray(ArrayName))
    for i in range(Volume.GetNumberOfPoints()):
        distance_=Volume.GetPointData().GetArray("signedDistanceFromWall").GetValue(i)
        pointId_ =Volume.GetPoint(i)
        velocity_=Velocity[i]
        #Store the data
        if distance_>DistanceThreshold:
            x.append(pointId_[0])
            y.append(pointId_[1])
            z.append(pointId_[2])
            u.append(velocity_[0])
            v.append(velocity_[1])
            w.append(velocity_[2])

    x=np.array(x); y=np.array(y); z=np.array(z)
    u=np.array(u); v=np.array(v); w=np.array(w)

    return x,y,z,u,v,w
                

def CardioPINNsGetVelocityData(velocity_path,VelocityArrayName,DistanceThresholdPercentile):
    #NOTE will need to normalize for future since velocity data is already normalized

    #Read the Velocity Data
    VelocityData=ReadVTUFile(velocity_path)

    #Compute Distance Away from The Wall
    DistanceFromWall=GetDistanceFromWall(VelocityData)
    DistanceFromWallNonZero=DistanceFromWall[DistanceFromWall!=0]

    #Add the Distance Array to VelocityFile
    VelocityData=PolyDataAddPointArray(VelocityData,DistanceFromWall,"signedDistanceFromWall")

    #Distance Away from the Wall
    DistanceThreshold=np.percentile(DistanceFromWallNonZero,DistanceThresholdPercentile)

    #Get Velocity Away From the Wall
    x,y,z,u,v,w=GetVelocityAwayFromWall(VelocityData,DistanceThreshold,VelocityArrayName)

    #Create a Dictionary to Assign As Input variables Constrains in PhysicsNemo
    data_invar={}
    data_invar["x"]=np.array([[x[i]] for i in range(len(x))])
    data_invar["y"]=np.array([[x[i]] for i in range(len(x))])
    data_invar["z"]=np.array([[x[i]] for i in range(len(x))])
    data_invar_numpy = {key: value for key, value in data_invar.items() if key in ["x", "y", "z"]}

    #Create a Dictionary to Assign As Output variables as Constrains in PhysicsNemo
    data_outvar={}
    data_outvar["u"]=np.array([[u[i]] for i in range(len(u))])
    data_outvar["v"]=np.array([[v[i]] for i in range(len(u))])
    data_outvar["w"]=np.array([[w[i]] for i in range(len(u))])
    data_outvar_numpy = {key: value for key, value in data_outvar.items() if key in ["u", "v", "w","p"]}

    data_outvar_numpy["continuity"] = np.zeros_like(data_outvar_numpy["u"])
    data_outvar_numpy["momentum_x"] = np.zeros_like(data_outvar_numpy["u"])
    data_outvar_numpy["momentum_y"] = np.zeros_like(data_outvar_numpy["u"])
    data_outvar_numpy["momentum_z"] = np.zeros_like(data_outvar_numpy["u"])

    return data_invar_numpy, data_outvar_numpy




############ Input/Output ##################
def ReadVTUFile(FileName):
	reader=vtk.vtkXMLUnstructuredGridReader()
	reader.SetFileName(FileName)
	reader.Update()
	return reader.GetOutput()

def ReadSTLFile(FileName):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(FileName)
    reader.Update()
    return reader.GetOutput()
    
def ReadVTKFile(FileName):
	reader = vtk.vtkStructuredPointsReader()
	reader.SetFileName(FileName)
	reader.ReadAllVectorsOn()
	reader.ReadAllScalarsOn()
	reader.Update()
	return reader.GetOutput()

def ReadVTPFile(FileName):
	reader=vtk.vtkXMLPolyDataReader()
	reader.SetFileName(FileName)
	reader.Update()
	return reader.GetOutput()

def ReadVTIFile(FileName):
	reader = vtk.vtkXMLImageDataReader() 
	reader.SetFileName(FileName) 
	reader.Update()
	return reader.GetOutput()

def WriteVTIFile(FileName,Data):
	writer=vtk.vtkXMLImageDataWriter()
	writer.SetFileName(FileName)
	writer.SetInputData(Data)
	writer.Update()

def WriteVTUFile(FileName,Data):
	writer=vtk.vtkXMLUnstructuredGridWriter()
	writer.SetFileName(FileName)
	writer.SetInputData(Data)
	writer.Update()
        
def WriteVTPFile(FileName,Data):
	writer=vtk.vtkXMLPolyDataWriter()
	writer.SetFileName(FileName)
	writer.SetInputData(Data)
	writer.Update()

def WritePolyDataFile(FileName,Data):
    writer=vtk.vtkPolyDataWriter()
    writer.SetFileName(FileName)
    writer.SetInputData(Data)
    writer.Update()

############# Mesh Morphing Functions ###############
        #Create a line from apex and centroid of the myocardium
        
def CreateLine(Point1,Point2,Length):
	line0=np.array([Point1[0]-Point2[0],Point1[1]-Point2[1],Point1[2]-Point2[2]])
	line1=-1*line0
	line0=(line0/np.linalg.norm(line0))*(Length/2.)
	line1=(line1/np.linalg.norm(line1))*(Length/2.)
	return line0,line1

def CreatePolyLine(Coords):
	# Create a vtkPoints object and store the points in it
	points = vtk.vtkPoints()
	for i in range(len(Coords)): points.InsertNextPoint(Coords[i])

	#Create a Polyline
	polyLine = vtk.vtkPolyLine()     
	polyLine.GetPointIds().SetNumberOfIds(len(Coords))
	for i in range(len(Coords)): polyLine.GetPointIds().SetId(i, i)

	# Create a cell array to store the lines in and add the lines to it
	cells = vtk.vtkCellArray()
	cells.InsertNextCell(polyLine)

	# Create a polydata to store everything in
	polyData = vtk.vtkPolyData()
    
	# Add the points to the dataset
	polyData.SetPoints(points)

	# Add the lines to the dataset
	polyData.SetLines(cells)
	
	return polyData 

def ClosestPoint(Point, Array):
	dist_2 = np.sum((Array - Point)**2, axis=1)
	return Array[np.argmin(dist_2)],np.argmin(dist_2),min(dist_2)

def FurthestPoint(Point, Array):
        dist_2 = np.sum((Array - Point)**2, axis=1)
        return Array[np.argmax(dist_2)],np.argmax(dist_2)

        
def CutPlane(Volume,Origin,Norm):
	plane=vtk.vtkPlane()
	plane.SetOrigin(Origin)
	plane.SetNormal(Norm)
	Slice=vtk.vtkCutter()
	Slice.GenerateTrianglesOff()
	Slice.SetCutFunction(plane)
	Slice.SetInputData(Volume)
	Slice.Update()
	return Slice.GetOutput()

def CutLine(Slice,Point,Centroid,Norm1):
	#Get the two in-plane normals
	Norm2_slice=(Point-Centroid)/np.linalg.norm(Point-Centroid)
	Norm3_slice=np.cross(Norm1,Norm2_slice)
	
	#Generate the two planes
	plane_N2=vtk.vtkPlane()
	plane_N2.SetOrigin(Centroid)
	plane_N2.SetNormal(Norm2_slice)
	plane_N3=vtk.vtkPlane()
	plane_N3.SetOrigin(Centroid)
	plane_N3.SetNormal(Norm3_slice)
	
	#Clip the plane to get a line across the diameter
	Line =vtk.vtkCutter()
	Line.GenerateTrianglesOff()
	Line.SetCutFunction(plane_N3)
	Line.SetInputData(Slice)
	Line.Update()
        
	#Separate the line into only one quarter (i.e. half the line)
	Line1=vtk.vtkClipPolyData()
	Line1.SetClipFunction(plane_N2)
	Line1.SetInputData(Line.GetOutput())
	Line1.Update()
	return Line1.GetOutput()

def CreateSphere(Coord,Radius):
	Sphere=vtk.vtkSphere()
	Sphere.SetCenter(Coord)
	Sphere.SetRadius(Radius)
	return Sphere

def GetCentroid(Surface):
	Centroid=vtk.vtkCenterOfMass()
	Centroid.SetInputData(Surface)
	Centroid.SetUseScalarsAsWeights(False)
	Centroid.Update()
	return Centroid.GetCenter()

def ComputeArea(Surface):
	masser = vtk.vtkMassProperties()
	masser.SetInputData(Surface)
	masser.Update()
	return masser.GetSurfaceArea()

def ExtractSurface(volume):
	#Get the outer surface of the volume
	surface=vtk.vtkDataSetSurfaceFilter()
	surface.SetInputData(volume)
	surface.Update()
	return surface.GetOutput()
        
#Print the progress of the loop
def PrintProgress(i,N,progress_old):
	progress_=(int((float(i)/N*100+0.5)))
	if progress_%10==0 and progress_%10!=progress_old: print ("    Progress: %d%%"%progress_)
	return progress_%10

def TagOuterSurface(Surface):
	#Create an OBB tree and cast Rays       
	obbTree = vtk.vtkOBBTree()
	obbTree.SetDataSet(Surface)
	obbTree.BuildLocator()
	pointsVTKintersection = vtk.vtkPoints()

	#Create an array to store surface tags
	Surface_tags=np.zeros(Surface.GetNumberOfPoints())
       
	#Get Centroid
	Centroid=np.array(GetCentroid(Surface))

	#Loop over all the points. 
	for i in range(Surface.GetNumberOfPoints()):
		pSource=np.array(Surface.GetPoint(i))
		pTarget=pSource+np.array((pSource-Centroid))*5
		code = obbTree.IntersectWithLine(pSource, pTarget, pointsVTKintersection, None)
		X=pointsVTKintersection.GetData().GetNumberOfTuples()
		if X>1: Surface_tags[i]=1
			
	#Store the data in Surface array
	#Tags for out or inner surface
	Surface_tags_vtk=numpy_to_vtk(Surface_tags,deep=True)
	Surface_tags_vtk.SetName("Tags")
	Surface.GetPointData().AddArray(Surface_tags_vtk)
	Surface.Modified()
                
	return Surface

#Smooth Surface
def SurfaceSmoothing(Surface,Nits,PB_value,method="Taubin"):
	if method=="Taubin":
		smoothingFilter = vtk.vtkWindowedSincPolyDataFilter()
		smoothingFilter.SetInputData(Surface)
		smoothingFilter.SetNumberOfIterations(Nits)
		smoothingFilter.SetPassBand(PB_value)
		smoothingFilter.SetBoundarySmoothing(True)
		smoothingFilter.Update()
		return smoothingFilter.GetOutput()
	elif method=="Laplace":
		smoothingFilter = vtk.vtkSmoothPolyDataFilter()
		smoothingFilter.SetInputData(Surface)
		smoothingFilter.SetNumberOfIterations(Nits)
		smoothingFilter.SetRelaxationFactor(PB_value)
		smoothingFilter.Update()
		return smoothingFilter.GetOutput()
	else:
		print ("Error. The smoothing filter was not found")
		exit(1)

def PolyDataAddPointArray(Surface,Array,ArrayName):
	SurfaceArray=numpy_to_vtk(Array,deep=True)
	SurfaceArray.SetName(ArrayName)
	Surface.GetPointData().AddArray(SurfaceArray)
	Surface.Modified()
	return Surface

def SurfaceAddCellArray(Surface,Array,ArrayName):
        SurfaceArray=numpy_to_vtk(Array,deep=True)
        SurfaceArray.SetName(ArrayName)
        Surface.GetCellData().AddArray(SurfaceArray)
        Surface.Modified()
        return Surface


def ProjectedPointOnLine(coord_,Centroid,Apex,Norm1):
	#Find the location (coord,distance) on the LV Apex-Base axis
	dist_P_to_line_=np.sqrt(vtk.vtkLine.DistanceToLine(coord_,Centroid,Apex))
	dist_P_to_Apex_=np.power( np.power(coord_[0]-Apex[0],2) + np.power(coord_[1]-Apex[1],2) + np.power(coord_[2]-Apex[2],2),0.5)
	dist_Apex_to_ProjP_=np.power(np.power(dist_P_to_Apex_,2)-np.power(dist_P_to_line_,2),0.5)
	coord_ProjP_=Apex-Norm1*dist_Apex_to_ProjP_

	return coord_ProjP_

def SurfaceNormals(Surface,FeatureAngle=None):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(Surface)
    if FeatureAngle is not None: normals.SetFeatureAngle(FeatureAngle)
    normals.AutoOrientNormalsOn()
    normals.UpdateInformation()
    normals.Update()
    Surface = normals.GetOutput()
    return Surface


def ThresholdByUpper(Volume,arrayname,value):
	Threshold=vtk.vtkThreshold()
	Threshold.SetInputData(Volume)
	Threshold.ThresholdByUpper(value)
	Threshold.SetInputArrayToProcess(0,0,0,"vtkDataObject::FIELD_ASSOCIATION_POINTS",arrayname)
	Threshold.Update()
	return Threshold.GetOutput()

def ThresholdInBetween(Volume,arrayname,value1,value2):
        Threshold=vtk.vtkThreshold()
        Threshold.SetInputData(Volume)
        Threshold.ThresholdBetween(value1,value2)
        Threshold.SetInputArrayToProcess(0,0,0,vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,arrayname)
        Threshold.Update()
        return Threshold.GetOutput()

def ConvertPointsToLine(PointsArray):
        # Create a vtkPoints object and store the points in it
        Points = vtk.vtkPoints()
        for Point_ in PointsArray:
                Points.InsertNextPoint(Point_)

        #Create a Polyline
        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(len(PointsArray))

        for i in range(0, len(PointsArray)):
                polyLine.GetPointIds().SetId(i, i)


        # Create a cell array to store the lines in and add the lines to it
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyLine)

        # Create a polydata to store everything in
        polyData = vtk.vtkPolyData()

        # Add the points to the dataset
        polyData.SetPoints(Points)

        # Add the lines to the dataset
        polyData.SetLines(cells)

        return polyData


def Statistics(Volume,ArrayName,NormalizationValue=None):
	statistics={"Volume":None, "Mean":None, "Stdev": None, "MeanNormalized":None, "StdevNormalized":None, "Median": None, "IQR":None, "Mode":None, "75thPerct": None, "Kurtosis": None, "Skewness":None, "Volume":None}

	#Convert VTK array to numpy
	Data_=vtk_to_numpy(Volume.GetPointData().GetArray(ArrayName))
	statistics["Mean"]      =np.mean(Data_)
	statistics["Stdev"]     =np.std(Data_)
	statistics["75thPerct"] =np.percentile(Data_,75)
	statistics["Median"]    =np.median(Data_)
	statistics["IQR"]       =IQR(Data_)
	if NormalizationValue is None:
		statistics["MeanNormalized"]=statistics["Mean"]/statistics["75thPerct"]
		statistics["StdevNormalized"]=statistics["Stdev"]/statistics["75thPerct"]
	else:
		statistics["MeanNormalized"]=statistics["Mean"]/NormalizationValue
		statistics["StdevNormalized"]=statistics["Stdev"]/NormalizationValue

	statistics["Skewness"]   =SKEWNESS(Data_)
	statistics["Kurtosis"]   =KURTOSIS(Data_)
	statistics["Mode"]       =MODE(Data_)[0]	
	

	Mass = vtk.vtkIntegrateAttributes()
	Mass.SetInputData(Volume)
	Mass.Update() 
	MassData=Mass.GetOutput()
	statistics["Volume"]=MassData.GetCellData().GetArray("Volume").GetValue(0)
	return statistics
		

def LargestConnectedRegion(Volume):
	ConnectedVolume=vtk.vtkConnectivityFilter()
	ConnectedVolume.SetInputData(Volume)
	ConnectedVolume.SetExtractionModeToLargestRegion()
	ConnectedVolume.Update()
	ConnectedVolumeData=ConnectedVolume.GetOutput()
	return ConnectedVolumeData
