import os
import argparse
import sys
from glob import glob
from utilities import ReadVTUFile,ReadVTPFile
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import copy

class ExtractVelocityAlongCenterline():
	def __init__(self,Args):
		self.Args=Args
		if self.Args.OutputFileName is None:
			directory_name=os.path.dirname(self.Args.InputFileName)
			if not self.Args.ComputeWSS:
				self.Args.OutputFileName=os.path.join(directory_name,os.path.basename(self.Args.InputFileName).split(".")[0]+"_CL_Velocity.dat")
				print ("\n---The output directory is: %s"%os.path.dirname(self.Args.OutputFileName))
				print ("---The output filename is:  %s"%os.path.basename(self.Args.OutputFileName))
			else:
				self.Args.OutputFileName=os.path.join(directory_name,os.path.basename(self.Args.InputFileName).split(".")[0]+"_CL_WSS.dat")
				print ("\n---The output directory is: %s"%os.path.dirname(self.Args.OutputFileName))
				print ("---The output filename is:  %s"%os.path.basename(self.Args.OutputFileName))
				

	def Main(self):
		#Read the Files
		if not self.Args.ComputeWSS:
			print ("\n--- Processing Input Velocity File: %s"%os.path.basename(self.Args.InputFileName))
			VelocityFile=ReadVTUFile(self.Args.InputFileName)	
			Coords=vtk_to_numpy(VelocityFile.GetPoints().GetData())*self.Args.ScalingFactor
		else:
			print ("\n--- Processing Input WSS File: %s"%os.path.basename(self.Args.InputFileName))
			WSSFile=ReadVTPFile(self.Args.InputFileName)
			Coords=vtk_to_numpy(WSSFile.GetPoints().GetData())*self.Args.ScalingFactor
	
		#Read the Surface file containing vmtkcenterlinemeshsections
		CenterlinesFile=ReadVTPFile(self.Args.CenterlinesFile)

		#Get the Coordinates from Centerlines File
		Npts=CenterlinesFile.GetNumberOfPoints()
		print ("\n---Number of Points in Centerline: %d"%Npts)
		PointsCL=np.zeros(shape=(Npts,3))
		MISR=np.zeros(Npts)
		for i in range(Npts):
			point_=CenterlinesFile.GetPoints().GetPoint(i)
			misr_=CenterlinesFile.GetPointData().GetArray("MaximumInscribedSphereRadius").GetValue(i)
			PointsCL[i,0]=point_[0]	
			PointsCL[i,1]=point_[1]	
			PointsCL[i,2]=point_[2]	
			MISR[i]=misr_	
	
		
		print ("\n--- Getting the Coordinates that are closest to CL")	
		if not self.Args.ComputeWSS:
			Velocity=[[] for i in range(Npts)]
			Pressure=[[] for i in range(Npts)]
		else:
			WSS=[[] for i in range(Npts)]


		#Loop over all the coordinates
		for i in range(len(Coords)):
			dist_ = np.sum((Coords[i] - PointsCL)**2, axis=1)
			dist_ = dist_**0.5
			idx_=np.argmin(dist_)
			#Get Velocity Magnitude and Store it
			if not self.Args.ComputeWSS:
				u_=VelocityFile.GetPointData().GetArray(self.Args.VelocityArrayName).GetValue(i*3)
				v_=VelocityFile.GetPointData().GetArray(self.Args.VelocityArrayName).GetValue(i*3+1)
				w_=VelocityFile.GetPointData().GetArray(self.Args.VelocityArrayName).GetValue(i*3+2)		
				mag_=(u_**2+v_**2+w_**2)**0.5
				p_=VelocityFile.GetPointData().GetArray(self.Args.PressureArrayName).GetValue(i)	
				#Store velocity and pressure
				Velocity[idx_].append(mag_)
				Pressure[idx_].append(p_)
			else:
				wss_u_=WSSFile.GetPointData().GetArray(self.Args.WSSArrayName).GetValue(i*3)	
				wss_v_=WSSFile.GetPointData().GetArray(self.Args.WSSArrayName).GetValue(i*3+1)	
				wss_w_=WSSFile.GetPointData().GetArray(self.Args.WSSArrayName).GetValue(i*3+2)	
				mag_=(wss_u_**2+wss_v_**2+wss_w_**2)**0.5
				WSS[idx_].append(mag_)

		

		outfile=open(self.Args.OutputFileName,'w')
		
		if not self.Args.ComputeWSS:
			outfile.write("X_CL Y_CL Z_CL Velocity_Mag Pressure Velocity_Stdev Pressure_Stdev\n")
			print ("---Writing Avergage Velocity and Pressure Along Centerline")
			for i in range(Npts):
				Avg_Velocity_=np.average(Velocity[i])
				Avg_Pressure_=np.average(Pressure[i])
				Stdev_Velocity_=np.std(Velocity[i])	
				Stdev_Pressure_=np.std(Pressure[i])
				outfile.write("%.05f %.05f %.05f %.05f %.05f %.05f %.05f\n"%(PointsCL[i,0],PointsCL[i,1],PointsCL[i,2],Avg_Velocity_,Avg_Pressure_,Stdev_Velocity_,Stdev_Pressure_))
			outfile.close()
			
		else:
			outfile.write("X_CL Y_CL Z_CL WSS_Mag WSS_Stdev\n")
			print ("--- Writing Average WSS Along Centerline")
			for i in range(Npts):
				Avg_WSS_=np.average(WSS[i])*self.Args.Viscosity
				Stdev_WSS_=np.std(WSS[i])*self.Args.Viscosity
				outfile.write("%.05f %.05f %.05f %.05f %.05f\n"%(PointsCL[i,0],PointsCL[i,1],PointsCL[i,2],Avg_WSS_,Stdev_WSS_))
			outfile.close()	




if __name__=="__main__":
        #Arguments
        
	parser= argparse.ArgumentParser(description="This script will extract velocity along centerline and also correct the velocity to have zero on the walls")

	parser.add_argument('-InputFileName', '--InputFileName', type=str, required=True, dest="InputFileName", help="Velocity File containing PINNs velocity")
	parser.add_argument('-CenterlinesFile', '--CenterlinesFile', type=str, required=True, dest="CenterlinesFile", help="File with Centerlines")
	parser.add_argument('-RadiusRatio', '--RadiusRatio', type=float, required=False, default=1.1, dest="RadiusRatio", help="Ratio of the maximum inscribed sphere radius to collect data (between 0 and 1).")
	parser.add_argument('-OutputFileName', '--OutputFileName', type=str, required=False, dest="OutputFileName", help="File name to store the data along centerline.")
	parser.add_argument('-VelocityArrayName', '--VelocityArrayName', type=str, required=False, default="Velocity_PINNs", dest="VelocityArrayName", help="Array name containing the velocity field.")
	parser.add_argument('-PressureArrayName', '--PressureArrayName', type=str, required=False, default="pressure_PINNs", dest="PressureArrayName", help="Array name containing the pressure field.")
	parser.add_argument('-WSSArrayName', '--WSSArrayName', type=str, required=False, default="WallShearRate", dest="WSSArrayName", help="Array name containing the WSS field.")
	parser.add_argument('-ComputeWSS', '--ComputeWSS', type=str, required=False, default=False, dest="ComputeWSS", help="Will look for an array called WSS.")
	parser.add_argument('-Viscosity', '--Viscosity', type=float, required=False, default=0.04, dest="Viscosity", help="Multiply Wall Shear Rate by visocsity of 0.04 dynes/cm2.")
	parser.add_argument('-ScalingFactor', '--ScalingFactor', type=float, required=False, default=1, dest="ScalingFactor", help="Scaling factor for coordinate of the surface/volume.")

	
        #Put all the arguments together
	args=parser.parse_args()

    	#Call your Class
	ExtractVelocityAlongCenterline(args).Main()

