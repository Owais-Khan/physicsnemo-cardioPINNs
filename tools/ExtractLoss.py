import argparse 
import os

class ExtractLoss():
	def __init__(self,Args):
		self.Args=Args
		if self.Args.OutputFileName is None:
			directory_name=os.path.dirname(self.Args.InputFileName)
			self.Args.OutputFileName=os.path.join(directory_name,os.path.basename(self.Args.InputFileName).split(".")[0]+"_Loss")
			print ("The output directory is: %s"%os.path.dirname(self.Args.OutputFileName))
			print ("The output filename is:  %s"%os.path.basename(self.Args.OutputFileName))


	def main(self):
		print ("Reading File: %s"%self.Args.InputFileName)
		infile=open(self.Args.InputFileName,'r')
		outfile=open(self.Args.OutputFileName,'w')
		outfile.write("Epoches Loss\n")	
		for LINE in infile:
			if LINE.find("loss:")>0: 
				loss_=float(LINE.split("loss:")[1].split(",")[0])
				epoches_=float(LINE.split("[step:")[1].split("]")[0])
				print ("---Epoches: %d, Loss: %.05f"%(epoches_,loss_))
				outfile.write("%d %.08f\n"%(epoches_,loss_))

		infile.close()
		outfile.close()
		
		

if __name__=="__main__":
        #Description
        parser = argparse.ArgumentParser(description="This script will write out the loss function to a separate file from simulation log file.")

        #Input filename of the perfusion map
        parser.add_argument('-InputFileName', '--InputFileName', type=str, required=True, dest="InputFileName",help="Log file from the simulation.")

        #Output argumenets
        parser.add_argument('-OutputFileName', '--OutputFileName', type=str, required=False, dest="OutputFileName",help="Output file with loss function")

        args=parser.parse_args()
        ExtractLoss(args).main() 
