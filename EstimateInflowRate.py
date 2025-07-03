import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

#This function finds an estimate of the inflow rate using training data.
#We find the closest velocity sample point from the inlet mesh (pointwise).
# We then estimate the flow rate from the velocity projection on the inlet mesh
#This is used to assign additional outlet BC to improve PINNs accuracy.

def EstimateInflowRate(inlet_mesh_vtk,inlet_centroid,inlet_normal_vector,velData_invar,velData_outvar,InletRadius,InletRadiusThreshold):
    Coords=vtk_to_numpy(inlet_mesh_vtk.GetPoints().GetData()) #inlet mesh coordinates
    Radius=np.zeros(inlet_mesh_vtk.GetNumberOfPoints()) #inlet mesh radius from center
    Velocity=np.zeros(inlet_mesh_vtk.GetNumberOfPoints()) #projected velocity onto inlet mesh
    DotValues=np.zeros(inlet_mesh_vtk.GetNumberOfPoints()) #Into or Out of the domain
    #Loop over all of the inlet coordinates
    for i in range(len(Coords)):
        point_=Coords[i]
        distance_=np.zeros(len(velData_invar["x"]))

        #Compute the radius
        Radius[i]=np.sqrt((Coords[i][0]-inlet_centroid[0])**2+(Coords[i][1]-inlet_centroid[1])**2+(Coords[i][2]-inlet_centroid[2])**2)

        #Loop over all the training data coordinates
        for j in range(len(velData_invar["x"])): 
            #Distance from inlet point to all training data points
            distance_[j]=np.sqrt((point_[0]-velData_invar["x"][j][0])**2+(point_[1]-velData_invar["y"][j][0])**2+(point_[2]-velData_invar["z"][j][0])**2)

        #Find the index of the point with the minimum distance
        min_idx_=np.argmin(distance_)

        #Velocity at this point
        u_=velData_outvar["u"][min_idx_][0]
        v_=velData_outvar["v"][min_idx_][0]
        w_=velData_outvar["w"][min_idx_][0]


        #Find the dot product of velocity and inlet normal vector (direction of velocity, to account for backward flow)
        dotvalue=np.dot(inlet_normal_vector,[u_,v_,w_])

        #Add Velocity Array
        Velocity[i]=np.sqrt(u_**2+v_**2+w_**2)

        #Determine whether flow in into or out of the domain
        if dotvalue>0: DotValues[i]=1 
        else: dotvalue:DotValues[i]=-1 

    
    #Sort the radius from center to outer
    RadiusSorted_indices=np.argsort(Radius)
    RadiusSorted=np.sort(Radius)
    VelocitySorted=Velocity[RadiusSorted_indices]
    DotValuesSorted=DotValues[RadiusSorted_indices]

    #Estimate max velocity assuming parabolic profile (vmax=v/(1-r^2/R^2))
    V_projected=[]
    Vmax_parabolic=[]
    for i in range(len(RadiusSorted)):
        if RadiusSorted[i]<=((InletRadiusThreshold/100.)*InletRadius):
            Vmax_parabolic.append((Velocity[i]*DotValues[i])/(1-RadiusSorted[i]**2/InletRadius**2))
            V_projected.append(Velocity[i]*DotValues[i])

    V=(InletRadiusThreshold/100.)*np.average(V_projected)+(1-InletRadiusThreshold/100.)*(np.average(Vmax_parabolic)/2.)

    InflowRate=np.pi*InletRadius**2*V
    return InflowRate
