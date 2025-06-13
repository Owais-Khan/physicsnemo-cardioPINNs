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


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    #------------------ Input Variables ------------------------------------------------
    nu=0.025 #viscosity
    inlet_vel=1.5  #inlet velocity
    NormalizeInput=True
    Scale = 0.4

    # ----------------- Read the STL Geometry Paths -------------------------------------
    point_path = to_absolute_path("./stl_files/Data2_aneurysm-mesh-complete/")
    
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

        #Temporary
        MeshCentroid=(-18.40381048596882, -50.285383353981196, 12.848136936899031)

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
    inlet_area=21.1284 * (Scale**2) #ComputeArea(inlet_mesh_vtk)
    outlet_area=[12.0773 * (Scale**2)]#[ComputeArea(outlet_mesh_vtk_) for outlet_mesh_vtk_ in outlet_mesh_vtk]
    
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

    """inlet_normal = (0.8526, -0.428, 0.299)
    inlet_area = 21.1284 * (scale**2)
    inlet_center = (-4.24298030045776, 4.082857101816247, -4.637790193399717)
    inlet_radius = np.sqrt(inlet_area / np.pi)
    outlet_normal = (0.33179, 0.43424, 0.83747)
    outlet_area = 12.0773 * (scale**2)
    outlet_radius = np.sqrt(outlet_area / np.pi)"""

    # make aneurysm domain
    domain = Domain()

    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nu * Scale, rho=1.0, dim=3, time=False)
    normal_dot_vel = NormalDotVec(["u", "v", "w"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    # add constraints to solver
    # inlet
    u, v, w = circular_parabola(
        Symbol("x"),
        Symbol("y"),
        Symbol("z"),
        center=inlet_centroid,
        normal=inlet_normal_vector,
        radius=inlet_radius,
        max_vel=inlet_vel,
    )
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_mesh,
        outvar={"u": u, "v": v, "w": w},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    for i in range(len(outlet_mesh)):
        outlet = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=outlet_mesh[i],
            outvar={"p": 0},
            batch_size=cfg.batch_size.outlet,
        )
        domain.add_constraint(outlet, "outlet%d"%i)

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=noslip_mesh,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.interior,
    )
    domain.add_constraint(interior, "interior")

    # Integral Continuity 1
    for i in range(len(outlet_mesh)):
        integral_continuity = IntegralBoundaryConstraint(
            nodes=nodes,
            geometry=outlet_mesh[i],
            outvar={"normal_dot_vel": 2.540},
            batch_size=1,
            integral_batch_size=cfg.batch_size.integral_continuity,
            lambda_weighting={"normal_dot_vel": 0.1},
        )
        domain.add_constraint(integral_continuity, "integral_continuity_%d"%i)

    # Integral Continuity 2
    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_mesh,
        outvar={"normal_dot_vel": -2.540},
        batch_size=1,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
    )
    domain.add_constraint(integral_continuity, "integral_continuity_2")

    # add validation data
    file_path = "./openfoam/aneurysm_parabolicInlet_sol0.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {
            "Points:0": "x",
            "Points:1": "y",
            "Points:2": "z",
            "U:0": "u",
            "U:1": "v",
            "U:2": "w",
            "p": "p",
        }
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)
        openfoam_invar = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y", "z"]
        }
        openfoam_invar = normalize_invar(openfoam_invar, center, scale, dims=3)
        openfoam_outvar = {
            key: value
            for key, value in openfoam_var.items()
            if key in ["u", "v", "w", "p"]
        }
        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar,
            true_outvar=openfoam_outvar,
            batch_size=4096,
        )
        domain.add_validator(openfoam_validator)
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials"
        )

    # add pressure monitor
    pressure_monitor = PointwiseMonitor(
        inlet_mesh.sample_boundary(16),
        output_names=["p"],
        metrics={"pressure_drop": lambda var: torch.mean(var["p"])},
        nodes=nodes,
    )
    domain.add_monitor(pressure_monitor)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
