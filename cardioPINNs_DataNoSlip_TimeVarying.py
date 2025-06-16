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


@physicsnemo.sym.main(config_path="conf", config_name="config_DataNoSlip")
def run(cfg: PhysicsNeMoConfig) -> None:

    #------------------ Input Variables ------------------------------------------------
    nu=0.025 #viscosity
    inlet_vel=1.5  #inlet velocity
    NormalizeInput=True
    Scale = 0.4
    DataLoss=True
    VelocityArrayName="velocity"
    DistanceThresholdPercentile=75

    # time window parameters
    time_window_size = 1.0
    t_symbol = Symbol("t")
    time_range = {t_symbol: (0, time_window_size)}
    nr_time_windows = 10

    # ----------------- Read the STL Geometry Paths -------------------------------------
    point_path = to_absolute_path("./stl_files/Data2_aneurysm-mesh-complete/")
    velocity_path = "/mnt/c/Users/owais/Research_Local/Simvascular_physicsnemo_cardioPINNs/aneurysm_scaled0.4/Simulations/aneurysm_scaled/results/all_results.vtu_01000.vtu"
    MeshPath="./stl_files/Data2_aneurysm-mesh-complete/mesh-complete.mesh.vtu"


    #Get the Paths for the Inflow surfaces
    inlet_path=glob(os.path.join(point_path,"inflow_new.stl"))[0]
    if len(inlet_path)==0: raise Exception("No inflow.stl file found. Exiting...")

    #Get the Paths for the Outflow surfaces
    outlet_path=sorted(glob(os.path.join(point_path,"cap_*new.stl")))
    if len(outlet_path)==0: raise Exception ("No cap_*.stl files found. Exiting...")
    else: print ("Number of Outlet Files: %d"%len(outlet_path))
    
    #Get the Path for the wall mesh
    wall_path=glob(os.path.join(point_path,"wall.stl"))[0]
    if len(wall_path)==0: raise Exception ("No wall.stl file found. Exiting...")

    #Get the Path for the encoled mesh
    meshcombined_path=glob(os.path.join(point_path,"mesh-combined.stl"))[0]
    if len(meshcombined_path)==0: raise Exception("No mesh-combined.stl found. Exiting...")

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

    #-------------------- Get Bounding Box Length -------------------------------------------
    BBox=interior_mesh_vtk.GetBounds()
    xRange=BBox[1]-BBox[0]
    yRange=BBox[3]-BBox[2]
    zRange=BBox[5]-BBox[4]
    normalizeRatio=np.sqrt(xRange**2+yRange**2+zRange**2)

    #------------------ Scaling Parameters for the Input Variables ----------------------------------------
    if (NormalizeInput is True):
        normalizeRatio=np.sqrt(xRange**2+yRange**2+zRange**2)
        if Scale is None: Scale=1/normalizeRatio
        MeshCentroid = tuple(GetCentroid(interior_mesh_vtk))

        #Normalize the inlet/outlet/interior meshes
        inlet_mesh = normalize_mesh(inlet_mesh, MeshCentroid, Scale)
        outlet_mesh = [normalize_mesh(outlet_mesh_, MeshCentroid, Scale) for outlet_mesh_ in outlet_mesh]
        noslip_mesh = normalize_mesh(noslip_mesh, MeshCentroid, Scale)
        integral_mesh = normalize_mesh(integral_mesh, MeshCentroid, Scale)
        interior_mesh = normalize_mesh(interior_mesh, MeshCentroid, Scale)

        #Normalize the inlet/outlet/interior meshes
        inlet_mesh_vtk = normalize_mesh_vtk(inlet_mesh_vtk, MeshCentroid, Scale)
        outlet_mesh_vtk = [normalize_mesh_vtk(outlet_mesh_, MeshCentroid, Scale) for outlet_mesh_ in outlet_mesh_vtk]
        noslip_mesh_vtk = normalize_mesh_vtk(noslip_mesh_vtk, MeshCentroid, Scale)
        integral_mesh_vtk = normalize_mesh_vtk(integral_mesh_vtk, MeshCentroid, Scale)
        interior_mesh_vtk = normalize_mesh_vtk(interior_mesh_vtk, MeshCentroid, Scale)

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


    print ("\n-------------------------")
    print ("Inlet Centroid: (%.05f, %.05f, %.05f)"%(inlet_centroid[0],inlet_centroid[1],inlet_centroid[2]))
    print ("Inlet Normal:   (%.05f, %.05f, %.05f)"%(inlet_normal_vector[0],inlet_normal_vector[1],inlet_normal_vector[2]))
    print ("Inlet Area:     %.05f"%inlet_area)
    for i in range(len(outlet_centroid)):
            print ("\n")
            print ("Outlet%d Centroid: (%.05f, %.05f, %.05f)"%(i,outlet_centroid[i][0],outlet_centroid[i][1],outlet_centroid[i][2]))
            print ("Outlet%d Normal:   (%.05f, %.05f, %.05f)"%(i,outlet_normal_vectors[i][0],outlet_normal_vectors[i][1],outlet_normal_vectors[i][2]))
            print ("Outlet%d Area:     %.05f"%(i,outlet_area[i]))

    # make aneurysm domain
    domain = Domain()

        
#---------------------------- Navier-Stokes ----------------------------------#
    #Navier-Stokes Solver
    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nu * Scale, rho=1.0, dim=3, time=True)
    normal_dot_vel = NormalDotVec(["u", "v", "w"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    
    time_window_net = MovingTimeWindowArch(flow_net, time_window_size)

    nodes = (ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")] + [time_window_net.make_node(name="time_window_network")] 
    )


#---------------------------- Initial Conditions -----------------------------#
    #make initial condition domain
    ic_domain = Domain("initial_conditions")

    # make moving window domain                                                                                                                                                                                                               
    window_domain = Domain("window")                                                                                                                                                                                                          
                                                                                                                                                                                                                                              
    # make initial condition                                                                                                                                                                                                                  
    ic = PointwiseInteriorConstraint(                                                                                                                                                                                                         
        nodes=nodes,                                                                                                                                                                                                                          
        geometry=interior_mesh, 
        outvar={"u": 0, "v":0, "w":0, "p":0,},                                                                                                                                                                                                    
        batch_size=cfg.batch_size.initial_condition,                                                                                                                                                                                          
        parameterization={t_symbol: 0},                                                                                                                                                                                                       
        )                                                                                                                                                                                                                                         
    ic_domain.add_constraint(ic, name="ic")   

                                                                                                                                                                                                                                              
    # make constraint for matching previous windows initial condition                                                                                                                                                                         
    ic = PointwiseInteriorConstraint(                                                                                                                                                                                                         
        nodes=nodes,                                                                                                                                                                                                                          
        geometry=interior_mesh,                                                                                                                                                                                                                         
        outvar={"u_prev_step_diff": 0, "v_prev_step_diff": 0, "w_prev_step_diff": 0},                                                                                                                                                         
        batch_size=cfg.batch_size.interior,                                                                                                                                                                                                   
        bounds=box_bounds,                                                                                                                                                                                                                    
        parameterization={t_symbol: 0},                                                                                                                                                                                                       
    )                                                                                                                                                                                                                                         
    window_domain.add_constraint(ic, name="ic")                                                                                                                                                                                               
                                                                                                                                                                                                                                              
    # make interior constraint                                                                                                                                                                                                                
    interior = PointwiseInteriorConstraint(                                                                                                                                                                                                   
        nodes=nodes,                                                                                                                                                                                                                          
        geometry=rec,                                                                                                                                                                                                                         
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},                                                                                                                                                          
        batch_size=4094,                                                                                                                                                                                                                      
        parameterization=time_range,                                                                                                                                                                                                          
    )                                                                                                                                                                                                                                         
    ic_domain.add_constraint(interior, name="interior")                                                                                                                                                                                       
    window_domain.add_constraint(interior, name="interior")                                    


    #Inlet Pressure
    inlet_pressure = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_mesh,
        outvar={"p": 0},
        batch_size=cfg.batch_size.inlet,
        )
    ic_domain.add_constraint(inlet_pressure, "inlet_pressure")
    window_domain.add_constraint(inlet_pressure, "inlet_pressure")

    # No-Slip on Wall
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=noslip_mesh,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
        parameterization=time_range,
    )
    ic_domain.add_constraint(no_slip, "no_slip")
    window_domain.add_constraint(no_slip,"no_slip")

    
    #Data Loss
    for i, specific_time in enumerate(np.linspace(0,time_window_size,10)):
        velData_invar, velData_outvar=CardioPINNsGetVelocityData(velocity_path,VelocityArrayName,DistanceThresholdPercentile)
        data = PointwiseConstraint.from_numpy(
            nodes=nodes,                                                                                                                              
            invar=velData_invar,                                                                                                                   
            outvar=velData_outvar,                                                                                                                 
            batch_size=cfg.batch_size.data,                                                                                                           
        )                                                                                                                                             
        domain.add_constraint(data, "data_constraints") 


    #---------------------- Add Output Validation Data ----------------------------                                                                                                                                                                                             
    # add inference data for time slices                                                                                                                                                                                                                                        
    for i, specific_time in enumerate(np.linspace(0, time_window_size, 10)):                                                                                                                                                                                                    
        vtk_obj=VTKFromFile(to_absolute_path("./stl_files/Data2_aneurysm-mesh-complete/mesh-complete.mesh.vtu"),export_map={"velocity": ["u", "v", "w"], "pressure": ["p"]},)

        grid_inference = PointVTKInferencer(                                                                                                                                                                                                                                    
            vtk_obj=vtk_obj,                                                                                                                                                                                                                                                    
            nodes=nodes,                                                                                                                                                                                                                                                        
            input_vtk_map={"x": "x", "y": "y", "z": "z"},                                                                                                                                                                                                                       
            output_names=["u", "v", "w", "p"],                                                                                                                                                                                                                                  
            requires_grad=False,                                                                                                                                                                                                                                                
            invar={"t": np.full([VolumetricMesh.GetNumberOfPoints(), 1], specific_time)}, 
            batch_size=100000,                                                                                                                                                                                                                                                  
        )                                                                                                                                                                                                                                                                       
        ic_domain.add_inferencer(grid_inference, name="time_slice_" + str(i).zfill(4))                                                                                                                                                                                          
        window_domain.add_inferencer(                                                                                                                                                                                                                                           
            grid_inference, name="time_slice_" + str(i).zfill(4)                                                                                                                                                                                                                
        )                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                

#----------------------------------- Add Monitors to Output ------------------------------

    # Inlet Pressure, Velocity and Flow Rates
    inlet_mesh_filename=os.path.splitext(os.path.basename(inlet_path))[0]
    inlet_monitor = PointwiseMonitor(
        inlet_mesh.sample_boundary(25),
        output_names=["u","v","w","p"],
        metrics={
            inlet_mesh_filename+"_pressure": lambda var: torch.mean(var["p"]),
            inlet_mesh_filename+"_flowrate": lambda var: inlet_area*torch.sum(torch.sqrt( torch.square(var["u"]) + torch.square(var["v"]) + torch.square(var["w"])))
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
                    mesh_filename_+"_pressure": lambda var: torch.mean(var["p"]),
                    mesh_filename_+"_flowrate": lambda var: outlet_area[i]*torch.sum(torch.sqrt( torch.square(var["u"]) + torch.square(var["v"]) + torch.square(var["w"])))
                    },
                nodes=nodes,
                )
        domain.add_monitor(outlet_monitor_)


    # monitors for the interior domain
    global_monitor = PointwiseMonitor(
        interior_mesh.sample_interior(100),
        output_names=["continuity", "momentum_x", "momentum_y", "momentum_z"],
        metrics={
            "mass_imbalance": lambda var: torch.sum(var["area"] * torch.abs(var["continuity"])),
            "momentum_imbalance": lambda var: torch.sum(var["area"]*(torch.abs(var["momentum_x"]) + torch.abs(var["momentum_y"])+torch.abs(var["momentum_z"]))),
        },
        nodes=nodes,
        requires_grad=True,
    )
    domain.add_monitor(global_monitor)



    #---------------------------------------- Start the Solver -------------------------------------
    # make solver                                                                                                                                                                                                                                                               
    slv = SequentialSolver(                                                                                                                                                                                                                                                     
        cfg,                                                                                                                                                                                                                                                                    
        [(1, ic_domain), (nr_time_windows, window_domain)],                                                                                                                                                                                                                     
        custom_update_operation=time_window_net.move_window,                                                                                                                                                                                                                    
    )                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                
             


if __name__ == "__main__":
    run()
