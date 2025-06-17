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


@physicsnemo.sym.main(config_path="conf", config_name="config_TimeVarying")
def run(cfg: PhysicsNeMoConfig) -> None:

    #------------------ Input Variables ------------------------------------------------
    nu=0.04 #viscosity assuming Mesh is in CGS units.
    cgsFactor=0.01 #multiply mesh/data by this factor to convert to cgs. Keep 1 by default.
    CenterInput=True #Normalize the input to enhance convergence
    DistanceThresholdPercentile=75 #How far away from wall to sample data

    #Do not touch. Work in progress
    MeshScale = 1.0 #Scaling factor for the mesh. If None, use Bounding Box*10 
    VelocityArrayName="Velocity" #Array name of the velocity in data files
    VelocityScale=None #Work in progress 

    # ----------------- Read the STL Geometry Paths -------------------------------------
    point_path = to_absolute_path("./stl_files/Data1_Stanford4DFlowMRI/")
    MeshPath=to_absolute_path("./stl_files/Data1_Stanford4DFlowMRI/mesh-complete.mesh.vtu")

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

    #Read the Velocity 
    #velocity_files=glob(os.path.join(velocity_path,"*.vtu"))
    #if len(velocity_files)==0: raise Exception("No velocity data found. Exiting...")
    #else: print ("Number of Velocity Files: %d"%len(velocity_files))


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

    #-------------------- Load and Normalize Velocity Data -----------------------------------
    #velocity_data=[ReadVTUFile(velocity_file_) for velocity_file_ in velocity_files]
    VelocityData=ReadVTUFile(VelocityFilePath)
    VelocityFileName=os.path.splitext(os.path.basename(VelocityFilePath))[0]
   
    #------------------- Convert to cgs -----------
    if (cgsFactor is not None) or (cgsFactor is not 1.0):
        print ("\n"+"-"*30)
        print ("Convert the Mesh to CGS units. CGS factor is: %.03f"%cgsFactor)
        MeshCentroid=(0,0,0)
        #Normalize the inlet/outlet/interior meshes
        inlet_mesh = normalize_mesh(inlet_mesh, MeshCentroid, cgsFactor)
        outlet_mesh = [normalize_mesh(outlet_mesh_, MeshCentroid, cgsFactor) for outlet_mesh_ in outlet_mesh]
        noslip_mesh = normalize_mesh(noslip_mesh, MeshCentroid, cgsFactor)
        integral_mesh = normalize_mesh(integral_mesh, MeshCentroid, cgsFactor)
        interior_mesh = normalize_mesh(interior_mesh, MeshCentroid, cgsFactor)

        #Normalize the inlet/outlet/interior meshes
        inlet_mesh_vtk = normalize_mesh_vtk(inlet_mesh_vtk, MeshCentroid, cgsFactor)
        outlet_mesh_vtk = [normalize_mesh_vtk(outlet_mesh_, MeshCentroid, cgsFactor) for outlet_mesh_ in outlet_mesh_vtk]
        noslip_mesh_vtk = normalize_mesh_vtk(noslip_mesh_vtk, MeshCentroid, cgsFactor)
        integral_mesh_vtk = normalize_mesh_vtk(integral_mesh_vtk, MeshCentroid, cgsFactor)
        interior_mesh_vtk = normalize_mesh_vtk(interior_mesh_vtk, MeshCentroid, cgsFactor)

        #Get Mesh Centroid and Bounding Box
        MeshCentroid = tuple(GetCentroid(interior_mesh_vtk))  
        BBox=interior_mesh_vtk.GetBounds()
        print ("--- New Centroid is:          (%.05f, %.05f, %.05f)"%(MeshCentroid[0],MeshCentroid[1],MeshCentroid[2]))
        print ("--- New X Bounds: (%.05f %.05f). Range=%.05f"%(BBox[0],BBox[1],BBox[1]-BBox[0]))
        print ("--- New Y Bounds: (%.05f %.05f). Range=%.05f"%(BBox[2],BBox[3],BBox[3]-BBox[2]))
        print ("--- New Z Bounds: (%.05f %.05f). Range=%.05f"%(BBox[4],BBox[5],BBox[5]-BBox[4]))

        #Convert the Velocity Mesh                                                                                                                                                                                                          
        VelocityDataNormalized=normalize_mesh_vtk(VelocityData,MeshCentroid, cgsFactor)

        #Normalize the mesh Inference Mesh
        Mesh=ReadVTUFile(MeshPath)
        MeshNormalized=normalize_mesh_vtk(Mesh,MeshCentroid,cgsFactor)
        WriteVTUFile("MeshNormalized.vtu",MeshNormalized)

    
    else:
        #Store original velocity data 
        VelocityDataNormalized=VelocityData

        #Store original mesh in output folder
        Mesh=ReadVTUFile(MeshPath)
        MeshNormalized=Mesh
        WriteVTUFile("MeshNormalized.vtu",Mesh)
            
    #------------------ Scaling Parameters for the Input Variables ----------------------------------------
    if (CenterInput is True):
        print ("\n"+"-"*30)
        print ("Centering the Input Data to Origin")
        MeshCentroid = tuple(GetCentroid(interior_mesh_vtk))

        #Normalize the inlet/outlet/interior meshes
        inlet_mesh = normalize_mesh(inlet_mesh, MeshCentroid, MeshScale)
        outlet_mesh = [normalize_mesh(outlet_mesh_, MeshCentroid, MeshScale) for outlet_mesh_ in outlet_mesh]
        noslip_mesh = normalize_mesh(noslip_mesh, MeshCentroid, MeshScale)
        integral_mesh = normalize_mesh(integral_mesh, MeshCentroid, MeshScale)
        interior_mesh = normalize_mesh(interior_mesh, MeshCentroid, MeshScale)

        #Normalize the inlet/outlet/interior meshes
        inlet_mesh_vtk = normalize_mesh_vtk(inlet_mesh_vtk, MeshCentroid, MeshScale)
        outlet_mesh_vtk = [normalize_mesh_vtk(outlet_mesh_, MeshCentroid, MeshScale) for outlet_mesh_ in outlet_mesh_vtk]
        noslip_mesh_vtk = normalize_mesh_vtk(noslip_mesh_vtk, MeshCentroid, MeshScale)
        integral_mesh_vtk = normalize_mesh_vtk(integral_mesh_vtk, MeshCentroid, MeshScale)
        interior_mesh_vtk = normalize_mesh_vtk(interior_mesh_vtk, MeshCentroid, MeshScale)

       #Get Mesh Centroid and Bounding Box
        MeshCentroid = tuple(GetCentroid(interior_mesh_vtk))
        BBox=interior_mesh_vtk.GetBounds()
        print ("--- New Centroid is:          (%.05f, %.05f, %.05f)"%(MeshCentroid[0],MeshCentroid[1],MeshCentroid[2]))
        print ("--- New X Bounds: (%.05f %.05f). Range=%.05f"%(BBox[0],BBox[1],BBox[1]-BBox[0]))
        print ("--- New Y Bounds: (%.05f %.05f). Range=%.05f"%(BBox[2],BBox[3],BBox[3]-BBox[2]))
        print ("--- New Z Bounds: (%.05f %.05f). Range=%.05f"%(BBox[4],BBox[5],BBox[5]-BBox[4]))

        #Normalize the Velocity Mesh
        VelocityDataNormalized=normalize_mesh_vtk(VelocityData,MeshCentroid, MeshScale)

        #Normalize the Inference Mesh
        MeshNormalized=normalize_mesh_vtk(MeshNormalized,MeshCentroid,MeshScale)
        WriteVTUFile("MeshNormalized.vtu",MeshNormalized)



    #----------------------------- Geometric Parame:ters -------------------------------
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

    #Navier-Stokes Solver
    # make list of nodes to unroll graph on
    print ("--- Creating Navier-Stokes Node...")
    ns = NavierStokes(nu=nu*MeshScale, rho=1.0, dim=3, time=False)

    #Normal Dot Vector
    #normal_dot_vel = NormalDotVec(["u", "v", "w"])

    #Flow Net
    print ("--- Flow Net Architecture...")
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,)


    print ("--- Putting all the nodes together...")
    nodes = (ns.make_nodes()
       # + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

#---------------------------- Initial Conditions -----------------------------#
    print ("\n"+"-"*30)
    print ("Creating Initial, Boudnary and Data Constraints")


    print ("--- Creating Interior Boundary Conditions (Continuity, Momentum)...")
    # Boundary Conditions for Interior                                                                                                                                                                                                       
    interior = PointwiseInteriorConstraint(                                                                                                                                                                                                   
        nodes=nodes,                                                                                                                                                                                                                          
        geometry=interior_mesh,                                                                                                                                                                                                          
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},                                                                                                                                                          
        batch_size=cfg.batch_size.interior,
        )            
    domain.add_constraint(interior, name="Interior_"+VelocityFileName)


    print ("--- Creating No-Slip Boundary Conditions on Wall ...") 
    #Boundary Conditions for the Wall
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=noslip_mesh,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
        )
    domain.add_constraint(no_slip, name="NoSlip_"+VelocityFileName)


    print ("--- Creating Zero-Pressure Boundary Condition at Inlet...") 
    #Boundary Conditions for Inlet Pressure
    inlet_pressure = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_mesh,
        outvar={"p": 0},
        batch_size=cfg.batch_size.inlet,
        )
    domain.add_constraint(inlet_pressure, name="InletPressure_"+VelocityFileName)


    #Data Loss 
    velData_invar, velData_outvar=CardioPINNsGetVelocityData(VelocityDataNormalized,VelocityArrayName,DistanceThresholdPercentile)
    data = PointwiseConstraint.from_numpy(
        nodes=nodes,                                                                                                                              
        invar=velData_invar,                                                                                                                   
        outvar=velData_outvar,                                                                                                                 
        batch_size=cfg.batch_size.data,                                                                                                           
        )                                                                                                                                             
    domain.add_constraint(data, "DataConstraints_"+VelocityFileName) 


#----------------------------------- Add Monitors to Output ------------------------------

    # Inlet Pressure, Velocity and Flow Rates
    inlet_mesh_filename=os.path.splitext(os.path.basename(inlet_path))[0]
    inlet_monitor = PointwiseMonitor(
        inlet_mesh.sample_boundary(25),
        output_names=["u","v","w","p"],
        metrics={
            inlet_mesh_filename+"_pressure_%s"%VelocityFileName: lambda var: torch.mean(var["p"]),
            inlet_mesh_filename+"_flowrate_%s"%VelocityFileName: lambda var: inlet_area*torch.sum(torch.sqrt(torch.square(var["u"]) + torch.square(var["v"]) + torch.square(var["w"])))
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
                    mesh_filename_+"_flowrate_%s"%VelocityFileName: lambda var: outlet_area[i]*torch.sum(torch.sqrt( torch.square(var["u"]) + torch.square(var["v"]) + torch.square(var["w"])))
                    },
                nodes=nodes,
                )
        domain.add_monitor(outlet_monitor_)


    # monitors for the interior domain
    """global_monitor = PointwiseMonitor(
        interior_mesh.sample_interior(100),
        output_names=["continuity", "momentum_x", "momentum_y", "momentum_z"],
        metrics={
            "mass_imbalance": lambda var: torch.sum(var["area"] * torch.abs(var["continuity"])),
            "momentum_imbalance": lambda var: torch.sum(var["area"]*(torch.abs(var["momentum_x"]) + torch.abs(var["momentum_y"])+torch.abs(var["momentum_z"]))),
        },
        nodes=nodes,
        requires_grad=True,
    )
    domain.add_monitor(global_monitor)"""

    # monitors for the interior domain                                                                                                                                                                                                        
    global_monitor = PointwiseMonitor(                                                                                                                                                                                                        
        interior_mesh.sample_interior(100),                                                                                                                                                                                                   
        output_names=["u", "v", "w", "p"],                                                                                                                                                                
        metrics={"InteriorVelocity": lambda var: torch.sum(torch.sqrt( torch.square(var["u"]) + torch.square(var["v"]) + torch.square(var["w"])))
        },                                                                                                                                                                                                                                    
        nodes=nodes,                                                                                                                                                                                                                          
        requires_grad=True,                                                                                                                                                                                                                   
    )                                                                                                                                                                                                                                         
    domain.add_monitor(global_monitor)      


    #---------------------- Add Output Validation Data ----------------------------
    vtk_obj = VTKFromFile("MeshNormalized.vtu",export_map={VelocityArrayName: ["u", "v", "w"], "pressure": ["p"]},)
    grid_inference = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y", "z": "z"},
        output_names=["u", "v","w", "p"],
        requires_grad=False,
        batch_size=1024,
    )
    domain.add_inferencer(grid_inference, VelocityFileName+".vtu")


    #---------------------------------------- Start the Solver -------------------------------------
    # make solver
    slv = Solver(cfg, domain)
    # start solver
    slv.solve()


if __name__ == "__main__":
    velocity_path = "/mnt/c/Users/owais/Research_Local/Simvascular_physicsnemo_cardioPINNs/Data1_Stanford4DFlowMRI_25Frames/"
    
    #Read the Velocity Files available in the folder
    velocity_files=glob(os.path.join(velocity_path,"*.vtu"))
    if len(velocity_files)==0: raise Exception("No velocity data found. Exiting...")
    else: print ("Number of Velocity Files: %d"%len(velocity_files))

    #Loop over all of the files
    for i in range(0,len(velocity_files)):
        VelocityFilePath=velocity_files[i]
        run()





