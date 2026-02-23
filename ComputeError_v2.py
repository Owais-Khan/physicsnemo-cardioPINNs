import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import argparse
from glob import glob
from utilities import *
import os

class ComputeErrors():
    def __init__(self,Args):
        self.Args=Args

    def main(self):
        GroundTruthVelFolder=sorted(glob(self.Args.InputFolder1+"/*_Rescaled.vtu")) #Ground truth velocity
        SimulatedVelFolder=sorted(glob(self.Args.InputFolder2+"/*_Rescaled.vtu")) #Simulated Velocity

        if len(GroundTruthVelFolder)!=len(SimulatedVelFolder):
            raise Exception("Number of Files in GroundTruth and Simulated Folders are not equal")
            exit(1)

        ErrorFile=open(self.Args.OutputFileName,'w')
        #Loop over the files
        for i in range(len(SimulatedVelFolder)):
            print ("Looping over Simulated Velocity File: %s"%SimulatedVelFolder[i])
            print ("--- Ground Truth Velocity File:       %s"%GroundTruthVelFolder[i])

            #Load Files
            GroundTruthVelVTK_= ReadVTUFile(GroundTruthVelFolder[i])
            SimulatedVelVTK_  =ReadVTUFile(SimulatedVelFolder[i])

            #Convert to numpy array
            GroundTruthVelArray_=vtk_to_numpy(GroundTruthVelVTK_.GetPointData().GetArray(self.Args.VelocityArrayName1))
            SimulatedVelArray_=vtk_to_numpy(SimulatedVelVTK_.GetPointData().GetArray(self.Args.VelocityArrayName2))

            GroundTruthVelMagSum_=np.sum(GroundTruthVelArray_[:,0]**2+GroundTruthVelArray_[:,1]**2+GroundTruthVelArray_[:,2]**2)
            print ("--- Sumation of Velocity Magnitude is: %.05f"%(GroundTruthVelMagSum_))
            VelErrorSum_=np.sum( (SimulatedVelArray_[:,0]-GroundTruthVelArray_[:,0])**2+(SimulatedVelArray_[:,1]-GroundTruthVelArray_[:,1])**2+(SimulatedVelArray_[:,2]-GroundTruthVelArray_[:,2])**2)
            print ("---- Sumation of GroundTruth - PINNs is: %.05f"%(VelErrorSum_))
            Error_=(VelErrorSum_/GroundTruthVelMagSum_)**0.5

            print ("--- L2 Norm of Error is: %.05f"%VelErrorSum_**0.5)

            ErrorFile.write("%s %.05f\n"%(os.path.basename(SimulatedVelFolder[i]),Error_))
            print ("--- Error is: %.05f"%Error_)
            print ("--- L2Norm is: %.05f"%VelErrorSum_**0.5)

        ErrorFile.close()


if __name__=="__main__":
        #Description
        parser = argparse.ArgumentParser(description="Computes the error between ground-truth and simulated data.")
        parser.add_argument('-InputFolder1', '--InputFolder1', required=True, dest="InputFolder1", help="Ground-truth velocity data.")
        parser.add_argument('-InputFolder2', '--InputFolder2', required=True, dest="InputFolder2",help="Simulated velocity data.")
        parser.add_argument('-VelocityArrayName1', '--VelocityArrayName1', required=False, default="Velocity_PINNs", dest="VelocityArrayName1",help="Array name where velocity data is stored for Grouth-truth File.")
        parser.add_argument('-VelocityArrayName2', '--VelocityArrayName2', required=False, default="Velocity_PINNs", dest="VelocityArrayName2",help="Array name where velocity data is stored for Simulated File.")
        parser.add_argument('-OutputFileName', '--OutputFileName', required=True, default="Errors.dat", dest="OutputFileName",help="File name where errors should be stored.")

        args=parser.parse_args()
        ComputeErrors(args).main()

