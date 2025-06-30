# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
from utilities import *
import torch
import numpy as np
from sympy import Symbol, sqrt, Max
import subprocess
import gc

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseConstraint,
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.geometry.primitives_3d import Box


from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.geometry.tessellation import Tessellation

from physicsnemo.sym.utils.io.vtk import VTKFromFile                                                                                                                                           
from physicsnemo.sym.domain.validator import PointVTKValidator
from physicsnemo.sym.domain.inferencer import PointVTKInferencer
from physicsnemo.sym.models.moving_time_window import MovingTimeWindowArch


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    #------------------ Input Variables ------------------------------------------------
    nu=0.04 #viscosity assuming Mesh is in CGS units.
    rho=1.06 # density in CGS units.
    cgsFactor=0.1 #multiply mesh/data by this factor to convert to cgs. Keep 1 by default.
    N_Files=25 #No of files
    Period=1.0 #seconds

    CenterInput=True #Normalize the input to enhance convergence
    DistanceThresholdPercentile=75 #How far away from wall to sample data
    InflowRate=241 #ml/s

    #Do not touch. Work in progress
    MeshScale = 1.0 #Scaling factor for the mesh. If None, use Bounding Box*10 
    VelocityArrayName="Velocity" #Array name of the velocity in data files
    VelocityScale=None #Work in progress 

    # ----------------- Read the STL Geometry Paths -------------------------------------
    point_path = to_absolute_path("./stl_files/Data1_Stanford4DFlowMRI/")
    MeshPath=to_absolute_path("./stl_files/Data1_Stanford4DFlowMRI/mesh-complete.mesh.vtu")
    velocity_path = "/mnt/c/Users/owais/Research_Local/Simvascular_physicsnemo_cardioPINNs/Data1_Stanford4DFlowMRI_25Frames/"
    
    print ("\n"+"-"*20)
    print ("Reading Surface Files for PINNs")
    #Get the Paths for the Inflow surfaces
    inlet_path=glob(os.path.join(point_path,"inflow.stl"))[0]
    if len(inlet_path)==0: raise Exception("No inflow.stl file found. Exiting...")

    #Get the Paths for the Outflow surfaces
    outlet_path=sorted(glob(os.path.join(point_path,"cap_*.stl")))
    if len(outlet_path)==0: raise Exception ("No cap_*.stl files found. Exiting...")
    else: print ("Number of Outlet Files: %d"%len(outlet_path))
    
    #Get the Path for the wall mesh
    wall_path=glob(os.path.join(point_path,"wall.stl"))[0]
    if len(wall_path)==0: raise Exception ("No wall.stl file found. Exiting...")

    #Get the Path for the encoled mesh
    meshcombined_path=glob(os.path.join(point_path,"mesh-combined.stl"))[0]
    if len(meshcombined_path)==0: raise Exception("No mesh-combined.stl found. Exiting...")

    #Get if the volumetric mesh is present
    if len(glob(MeshPath))==0: raise Exception("No mesh-complete.mesh.vtu found. Exiting..")

    #-------------------- Load the STL Mesh into PhysicsNeMo --------------------------------
    inlet_mesh     = Tessellation.from_stl(inlet_path, airtight=False)
    outlet_mesh    = [Tessellation.from_stl(outlet_path_, airtight=False) for outlet_path_ in outlet_path]
    noslip_mesh    = Tessellation.from_stl(wall_path, airtight=False)
    integral_mesh = Tessellation.from_stl(inlet_path, airtight=False)
    interior_mesh = Tessellation.from_stl(meshcombined_path, airtight=True)

    #-------------------- Load the STL Mesh in VTK ------------------------------------------
    inlet_mesh_vtk    = ReadSTLFile(inlet_path)
    outlet_mesh_vtk   = [ReadSTLFile(outlet_path_) for outlet_path_ in outlet_path]
    noslip_mesh_vtk   = ReadSTLFile(wall_path)
    integral_mesh_vtk = ReadSTLFile(inlet_path)
    interior_mesh_vtk = ReadSTLFile(meshcombined_path)

    #------------------- Convert to cgs units and center  to zero -----------
    print ("\n"+"-"*30)
    print ("Convert the Mesh to CGS units. CGS factor is: %.03f"%cgsFactor)
    MeshCentroidOld= tuple(GetCentroid(interior_mesh_vtk))  
    BBoxOld=GetBoundingBox(interior_mesh_vtk)

    #Center the model to origin. Apply cgs factor (physics-nemo)
    inlet_mesh = normalize_mesh(inlet_mesh, MeshCentroidOld, cgsFactor)
    outlet_mesh = [normalize_mesh(outlet_mesh_, MeshCentroidOld, cgsFactor) for outlet_mesh_ in outlet_mesh]
    noslip_mesh = normalize_mesh(noslip_mesh, MeshCentroidOld, cgsFactor)
    integral_mesh = normalize_mesh(integral_mesh, MeshCentroidOld, cgsFactor)
    interior_mesh = normalize_mesh(interior_mesh, MeshCentroidOld, cgsFactor)


    #Center the model to origin. Apply cgs factor (vtk)
    inlet_mesh_vtk = normalize_mesh_vtk(inlet_mesh_vtk, MeshCentroidOld, cgsFactor)
    outlet_mesh_vtk = [normalize_mesh_vtk(outlet_mesh_, MeshCentroidOld, cgsFactor) for outlet_mesh_ in outlet_mesh_vtk]
    noslip_mesh_vtk = normalize_mesh_vtk(noslip_mesh_vtk, MeshCentroidOld, cgsFactor)
    integral_mesh_vtk = normalize_mesh_vtk(integral_mesh_vtk, MeshCentroidOld, cgsFactor)
    interior_mesh_vtk = normalize_mesh_vtk(interior_mesh_vtk, MeshCentroidOld, cgsFactor)

    #Get Mesh Centroid and Bounding Box
    MeshCentroidNew = tuple(GetCentroid(interior_mesh_vtk)) 
    BBoxNew=GetBoundingBox(interior_mesh_vtk)
    print ("--- Old Centroid is: (%.05f, %.05f, %.05f)"%(MeshCentroidOld[0],MeshCentroidOld[1],MeshCentroidOld[2]))
    print ("--- Old X Bounds: (%.05f %.05f). Range=%.05f"%(BBoxOld[0],BBoxOld[1],BBoxOld[1]-BBoxOld[0]))
    print ("--- Old Y Bounds: (%.05f %.05f). Range=%.05f"%(BBoxOld[2],BBoxOld[3],BBoxOld[3]-BBoxOld[2]))
    print ("--- Old Z Bounds: (%.05f %.05f). Range=%.05f"%(BBoxOld[4],BBoxOld[5],BBoxOld[5]-BBoxOld[4]))
    print ("\n")
    print ("--- New Centroid is: (%.05f, %.05f, %.05f)"%(MeshCentroidNew[0],MeshCentroidNew[1],MeshCentroidNew[2]))
    print ("--- New X Bounds: (%.05f %.05f). Range=%.05f"%(BBoxNew[0],BBoxNew[1],BBoxNew[1]-BBoxNew[0]))
    print ("--- New Y Bounds: (%.05f %.05f). Range=%.05f"%(BBoxNew[2],BBoxNew[3],BBoxNew[3]-BBoxNew[2]))
    print ("--- New Z Bounds: (%.05f %.05f). Range=%.05f"%(BBoxNew[4],BBoxNew[5],BBoxNew[5]-BBoxNew[4])) 

    #----------------------------- Geometric Parameters -------------------------------
    #Surface Normals
    WallNormals=SurfaceNormals(interior_mesh_vtk)
    
    #Get normal vectors and centroid for the inlet
    inlet_centroid,inlet_centroid_id,min_dist_=ClosestPoint(GetCentroid(inlet_mesh_vtk),vtk_to_numpy(inlet_mesh_vtk.GetPoints().GetData())) #Closest point to the centroid, id, min distance
    inlet_normal=ProjectData(SourceMesh=WallNormals,InputMesh=inlet_mesh_vtk)
    inlet_normal_vector=np.array(vtk_to_numpy(inlet_normal.GetPointData().GetArray("Normals"))[inlet_centroid_id])*-1 #Get centroid normal and flip it for inward flow.

    #Get normal vectors and centroid for the outlets
    outlet_centroid=[]
    outlet_normal_vectors=[]
    for i in range(len(outlet_mesh_vtk)):
        centroid_,centroidID_,min_dist_=ClosestPoint(GetCentroid(outlet_mesh_vtk[i]),vtk_to_numpy(outlet_mesh_vtk[i].GetPoints().GetData())) #Closest point to the centroid and id
        outlet_normal_=ProjectData(SourceMesh=WallNormals,InputMesh=outlet_mesh_vtk[i])
        outlet_normal_vector_=np.array(vtk_to_numpy(outlet_normal_.GetPointData().GetArray("Normals"))[centroidID_]) #Get centroid normal
        outlet_centroid.append(centroid_)
        outlet_normal_vectors.append(outlet_normal_vector_)

    #Get Surface Areas
    inlet_area=ComputeArea(inlet_mesh_vtk)
    outlet_area=[ComputeArea(outlet_mesh_vtk_) for outlet_mesh_vtk_ in outlet_mesh_vtk]
    
    #Get Radii
    inlet_radius=np.sqrt(inlet_area/np.pi)
    outlet_radius=[np.sqrt(outlet_area_/np.pi) for outlet_area_ in outlet_area]


    print ("\n"+"-"*30)
    print ("Inlet Centroid: (%.05f, %.05f, %.05f)"%(inlet_centroid[0],inlet_centroid[1],inlet_centroid[2]))
    print ("Inlet Normal:   (%.05f, %.05f, %.05f)"%(inlet_normal_vector[0],inlet_normal_vector[1],inlet_normal_vector[2]))
    print ("Inlet Area:     %.05f"%inlet_area)
    for i in range(len(outlet_centroid)):
            print ("\n")
            outlet_filename_=os.path.basename(outlet_path[i])
            print ("%s Centroid: (%.05f, %.05f, %.05f)"%(outlet_filename_,outlet_centroid[i][0],outlet_centroid[i][1],outlet_centroid[i][2]))
            print ("%s Normal:   (%.05f, %.05f, %.05f)"%(outlet_filename_,outlet_normal_vectors[i][0],outlet_normal_vectors[i][1],outlet_normal_vectors[i][2]))
            print ("%s Area:     %.05f"%(outlet_filename_,outlet_area[i]))

    # make aneurysm domain
    domain = Domain()

#---------------------------- Navier-Stokes ----------------------------------#
    print ("\n"+"-"*30)
    print ("Creating Network Architecture...")

    # time window parameters
    t_symbol = Symbol("t")
    time_range = {t_symbol: (0, Period)}

    #Navier-Stokes Solver
    # make list of nodes to unroll graph on
    print ("--- Creating Navier-Stokes Node...")
    ns = NavierStokes(nu=nu, rho=1.06, dim=3, time=True)

    #Normal Dot Vector
    normal_dot_vel = NormalDotVec(["u", "v", "w"])

    #Flow Net
    print ("--- Flow Net Architecture...")
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"),Key("t")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,)


    print ("--- Putting all the nodes together...")
    nodes = (ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    #make domain
    domain=Domain()


#---------------------------- Constraints -----------------------------#
    print ("\n"+"-"*30)
    print ("Creating Initial, Boudnary and Data Constraints")

    """print ("--- Creating Inlet Dirichlet Boundary Condition...")
    PeakInletVel=(2*(InflowRate/inlet_area))
    print ("\n------ Assigned Flow Rate at %s: %.05f"%(os.path.splitext(os.path.basename(inlet_path))[0],InflowRate))
    print ("--------- Peak Velocity is:        %.05f"%PeakInletVel)
    print ("--------- Peak Reynolds # is:      %.05f"%((rho*(PeakInletVel*0.5)*(2*inlet_radius))/nu))
    u, v, w = circular_parabola(
        Symbol("x"),
        Symbol("y"),
        Symbol("z"),
        center=inlet_centroid,
        normal=inlet_normal_vector,
        radius=inlet_radius,
        max_vel=2*(InflowRate/inlet_area),)
    
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_mesh,
        outvar={"u": u, "v": v, "w": w},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet, "Dirichlet_Inlet")


    print ("\n--- Creating Integral Boundary Condition at Inlet...")
    # Integral Continuity 1                                                                                                                                 
    print ("------ Assigned Flow Rate at Inlet: %.05f"%InflowRate)
    integral_continuity = IntegralBoundaryConstraint( 
        nodes=nodes,
        geometry=inlet_mesh,                  
        outvar={"normal_dot_vel": -1*InflowRate},
        batch_size=1,
        integral_batch_size=cfg.batch_size.integral_continuity,                                    
        lambda_weighting={"normal_dot_vel": 0.1},
        )                      
    domain.add_constraint(integral_continuity, "Integral_Inlet") 
                                                                                                                                                                                                                                              
    # Integral Continuity 2                                                                                                                                                                                      
    print ("\n--- Creating Integral Boundary Condition at Outlets Using Area Ratios...")
    for i in range(len(outlet_mesh)):
        flow_rate_=(outlet_area[i]/np.sum(outlet_area))*InflowRate
        print ("------ Assigned Flow Rate at %s: %.05f"%(os.path.splitext(os.path.basename(outlet_path[i]))[0],flow_rate_))
        integral_continuity = IntegralBoundaryConstraint( 
            nodes=nodes,                                                                                                                           
            geometry=outlet_mesh[i],                                                                                                               
            outvar={"normal_dot_vel": flow_rate_},              
            batch_size=1,
            integral_batch_size=cfg.batch_size.integral_continuity, 
            lambda_weighting={"normal_dot_vel": 0.1},
        )                                                                                                                                                          
        domain.add_constraint(integral_continuity, "Integral_%s"%os.path.splitext(os.path.basename(outlet_path[i]))[0])"""



    print ("--- Creating Interior Boundary Conditions (Continuity, Momentum)...")
    interior = PointwiseInteriorConstraint(                                                                                         
        nodes=nodes,                                                                                                                
        geometry=interior_mesh,       
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},    
        batch_size=cfg.batch_size.interior,
        compute_sdf_derivatives=True,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            "momentum_z": Symbol("sdf")},
        parameterization=time_range,
        )            
    domain.add_constraint(interior, name="Interior")

    print ("--- Creating No-Slip Boundary Conditions on Wall ...") 
    #Boundary Conditions for the Wall
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=noslip_mesh,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
        parameterization=time_range,
        )
    domain.add_constraint(interior, name="NoSlip")

    print ("--- Creating Zero-Pressure Boundary Condition at Inlet...") 
    #Boundary Conditions for Inlet Pressure
    inlet_pressure = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_mesh,
        outvar={"p": 0},
        batch_size=cfg.batch_size.inlet,
        parameterization=time_range,
        )
    domain.add_constraint(interior, name="InletPressure")


    #Read All of the Velocity Data
    print ("\n"+"-"*30)
    velocity_files=sorted(glob(os.path.join(velocity_path,"*.vtu")))
    if len(velocity_files)==0: raise Exception("No velocity data found. Exiting...")
    else: print ("Number of Velocity Files: %d"%len(velocity_files))
    VelocityData=[normalized_mesh_vtk(ReadVTUFile(filename_),MeshCentroidOld,cgsFactor) for filename_ in velocity_files] 
    #Create symbolic represntation of the data
    U=[];V=[];W=[]
    for i in range(len(velocity_files)):
        u_,v_,w_=ProbeVelocityData(Symbol("x"),Symbol("y"),Symbol("z"),Data=VelocityData[i])
        U.append(u_)
        V.append(v_)
        W.append(w_)

    #Create point constraint for each of the time step
    Data_Constraints=[]
    timeArray=np.linspace(0,Period,len(VelocityData))
    for i in range(len(VelocityData)):
        dataConstraint_=PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=interior_mesh,
            outvar={"u": U[i], "v": V[i], "w": W[i]},
            batch_size=cfg.batch_size.interior_mesh,
            parameterization=time_range,
        )
        domain.add_constraint(DataConstraint, "DataConstraint_%d"%i)


#----------------------------------- Add Monitors to Output ------------------------------
    """# Inlet Pressure, Velocity and Flow Rates
    inlet_mesh_filename=os.path.splitext(os.path.basename(inlet_path))[0]
    inlet_monitor = PointwiseMonitor(
        inlet_mesh.sample_boundary(25),
        output_names=["u","v","w","p"],
        metrics={
            inlet_mesh_filename+"_pressure_%s"%VelocityFileName: lambda var: torch.mean(var["p"]),
            inlet_mesh_filename+"_flowrate_%s"%VelocityFileName: lambda var: inlet_area*torch.mean(torch.sqrt(torch.square(var["u"]) + torch.square(var["v"]) + torch.square(var["w"])))
            },
        nodes=nodes,
    )
    domain.add_monitor(inlet_monitor)

    #Outlet Pressure, Velocity and Flow Rates
    for i in range(len(outlet_mesh)):
        mesh_filename_=os.path.splitext(os.path.basename(outlet_path[i]))[0]
        outlet_monitor_= PointwiseMonitor(
                outlet_mesh[i].sample_boundary(25),
                output_names=["u","v","w","p"],
                metrics={
                    mesh_filename_+"_pressure_%s"%VelocityFileName: lambda var: torch.mean(var["p"]),
                    mesh_filename_+"_flowrate_%s"%VelocityFileName: lambda var: outlet_area[i]*torch.mean(torch.sqrt( torch.square(var["u"]) + torch.square(var["v"]) + torch.square(var["w"])))
                    },
                nodes=nodes,
                )
        domain.add_monitor(outlet_monitor_)


    # monitors for the interior domain 
    global_monitor = PointwiseMonitor(
            interior_mesh.sample_interior(100),                                                                                                                                                                                 
            output_names=["u", "v", "w", "p"],
            metrics={"InteriorVelocity": lambda var: torch.mean(torch.sqrt( torch.square(var["u"]) + torch.square(var["v"]) + torch.square(var["w"])))},                                                                       
            nodes=nodes,     
        requires_grad=True,                                                                                                                                                                                                                   
    )                                                                                                                                                                                                                                         
    domain.add_monitor(global_monitor)      


    #---------------------- Add Output Validation Data ----------------------------
    vtk_obj = VTKFromFile(os.path.join(VelocityFilePath),export_map={VelocityArrayName+"_PINNs": ["u", "v", "w"], "pressure_PINNs": ["p"]},)
    points = vtk_obj.get_points()
    for i in range(3): points[:, i] =(points[:,i] - MeshCentroidOld[i])*cgsFactor  # Scale the data
    vtk_obj.set_points(points)
    grid_inference = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y", "z": "z"},
        output_names=["u", "v","w", "p"],
        requires_grad=False,
        batch_size=1024,
    )
    domain.add_inferencer(grid_inference, VelocityFileName+".vtu")"""


    #---------------------------------------- Start the Solver -------------------------------------
    # make solver
    slv = Solver(cfg, domain)
    # start solver
    slv.solve()



    """#---------------------------- Rescale the Inferenced Mesh -------------------------------------
    #Renormalize the Inference File
    print ("\n"+"-"*30)
    print ("ReScaling the Velocity Data: %s.vtu"%VelocityFileName)
    VelocityDataPINNs=ReadVTUFile(os.path.join("inferencers/%s.vtu"%VelocityFileName))
    VelocityDataPINNsRescaled=reverse_normalize_mesh_vtk(VelocityDataPINNs,MeshCentroidOld,cgsFactor)
    WriteVTUFile(os.path.join("inferencers/%s_Rescaled.vtu"%VelocityFileName),VelocityDataPINNsRescaled)

    #--------------------------- Compute Data Loss ------------------------------------------------
    print ("\n"+"-"*30)
    print ("Storing Data Error in: inferencers/DataLoss.dat")
    DataLossFile=open(os.path.join("inferencers/DataLoss.dat"),'a')
    velData_invar_PINNs, velData_outvar_PINNs=CardioPINNsGetVelocityData(VelocityDataPINNs,VelocityArrayName+"_PINNs",DistanceThresholdPercentile) #Extract the Same Points as used in DataLoss
    
    #Compute Error 
    VelocityMagnitudeSum=torch.sum(torch.square(torch.tensor(velData_outvar["u"]))  + torch.square(torch.tensor(velData_outvar["v"])) + torch.square(torch.tensor(velData_outvar["w"])))
    VelocityErrorSum=torch.square(torch.tensor(velData_outvar["u"])-torch.tensor(velData_outvar_PINNs["u"]))
    VelocityErrorSum+=torch.square(torch.tensor(velData_outvar["v"])-torch.tensor(velData_outvar_PINNs["v"]))
    VelocityErrorSum+=torch.square(torch.tensor(velData_outvar["w"])-torch.tensor(velData_outvar_PINNs["w"]))
    VelocityErrorSum=torch.sum(VelocityErrorSum)

    VelocityError=(VelocityErrorSum/VelocityMagnitudeSum)**0.5
    DataLossFile.write("%s %.05f\n"%(VelocityFileName,VelocityError))
    DataLossFile.close()
    print ("\n\n\n"+"-"*30)"""
    
    #Delete the solver for next timestep
    del flow_net, nodes, domain, interior, vtk_obj, no_slip, inlet_pressure, data, integral_continuity, inlet_monitor, outlet_monitor_, global_monitor, grid_inference, slv 

    #Collect garbage
    gc.collect()

    #collect pytorch's cuda chache
    torch.cuda.empty_cache()

if __name__ == "__main__":

    
    run()
