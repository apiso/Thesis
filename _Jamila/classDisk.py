#FILE: classDisk
#FUNCTION: This file contains the class Disk, which is meant to simulate a protoplanetary disk with particular density and temperature profiles and given molecular species.


######
###BACKGROUND
#Below Section: Imports necessary functions
import numpy as np
import matplotlib.pyplot as graph
import astropy.constants as const

import diskutils as utils
from classMol import Mol



class Disk(object):
#FUNCTION: __init__
#PURPOSE: This function is meant to initialize a disk with given temperature and density profiles.
	def __init__(self, mols=None):
		#Below Section: Initializes molecular species in disk
		if mols == None: #If no mols given
			self.mols = []
		elif mols != None: #If list of mols given
			self.mols = mols
			
			

#FUNCTION: addMol
#PURPOSE: This function is meant to add a mol to the current disk.
	def addMol(self, thismol):
		#Below Section: Adds given mol to disk's mols
		if thismol not in self.mols: #If not in disk yet
			self.mols.append(thismol)
		elif thismol in self.mols: #If mol already in disk
			print("This molecular species is already in the disk.  Nothing was changed.")
			
			

#FUNCTION: delMol
#PURPOSE: This function is meant to delete a mol from the current disk.
	def delMol(self, thismol):
		#Below Section: Deletes given mol from disk's mols
		if thismol in self.mols: #If in disk
			self.mols.remove(thismol)
		elif thismol not in self.mols: #If mol not in disk
			print("This molecular species was not in the disk to start with.  Nothing was changed.")
		

	
#FUNCTION: printMol
#PURPOSE: This function is meant to return a string listing the mols currently contained in the disk.
	def stringMol(self):
		#Below Section: Builds a string of molecular species within the disk
		names = '['
		for molhere in self.mols: #Iterates through mols
			names = names + molhere.name + ', '
		names = names[0:-2] + ']'	
		
		#Below Section: Returns finished string
		return names
		
		
		
#FUNCTION: profT
#PURPOSE: This function is meant to calculate a temperature profile for the given disk using the given type.
	def profT(self, R, z, type='basic'):
		#Below Section: Calls the given type of profile
		if type == 'basic': #Basic profile
			return utils.tempprofbasic(R=R)
			
		elif type == 'cos': #Cosine profile
			Tmid = utils.tempprofbasic(R=R, z=0.0)
			done = utils.tempprofcos(Tmid=Tmid, R=R, z=z)
			return done

		
		
#FUNCTION: profnH
#PURPOSE: This function is meant to calculate a density profile for the given disk using the given type.
	def profnH(self, R, z, mu=1.0, mstar=1.0, type='basic'):
		#Below Section: Calls the given type of profile
		if type == 'basic': #Basic profile
			return utils.densprofbasic(R=R)
			
		elif type == 'gaus-basic': #Gaussian profile
			T = utils.tempprofbasic(R=R, z=z)
			return utils.densprofgaus(R=R, z=z, T=T, mu=mu, mstar=mstar)
			#Requires R, z, mu, mstar, T, N
			
		elif type == 'gaus-cos': #Gaussian profile with cos T
			densmatr = np.zeros(shape=(len(z), len(R)))
			for p in range(0, len(z)):
				Tmid = utils.tempprofbasic(R=R, z=0.0)
				T2d = utils.tempprofcos(R=R, z=z[p], mu=mu, mstar=mstar, Tmid=Tmid)
				densmatr[p, 0:len(R)] = utils.densprofgaus(R=R, z=z[p], T=T2d, mu=mu, mstar=mstar)
			return densmatr
			
			

			
#FUNCTION: getsnowlines
#PURPOSE: This function iterates through the molecules contained within the current disk and produces the radii (AU) of their snow line locations.
	def getsnowlines(self, R, mollist,
					z=None, profTtype='basic',
					profnHtype='basic'):
		#Below Section: Iterates through molecules in disk
		radvals = [] #For holding x-value locations
		radnames = [] #For holding names relating to x-values
		ntot = self.profnH(R=R, z=z, type=profnHtype) #Base nH
		molstuff = self.mols
		for molhere in self.mols:
			#Below skips current mol if not in list of desired snow lines
			if molhere.name not in mollist:
				continue
			
			#Below calculates density for current molecule
			denshere = utils.calcnmol(R=R, z=z, disk=self, mol=molhere, profTtype=profTtype, profnHtype=profnHtype)[0]/1.0/ntot
			normdens = denshere/1.0/max(denshere[~np.isnan(denshere)])
			normdens[np.isnan(normdens)] = 0
			
			#Below finds closest y-value to .5 in normalized density for detecting snow line
			setval = .5
			nearstuffhere = utils.findnearest(normdens, setval)
			nearvalhere = nearstuffhere['value']
			nearindhere = nearstuffhere['index']
						
			if np.isnan(nearvalhere):
				#print("nan encountered for snow line at "+molhere.name+". Skipping this snow line.")
				continue
			
			#Below makes sure points not too far away
			if abs(setval - nearvalhere) > .02:
				continue
				#raise ValueError("Looks like no points are close enough to the snow line for "+molhere.name+".  Please input an array of radii with smaller spacing or more points.")
			
			#Below records the current radii with others
			radvals.append(R[nearindhere])
			radnames.append(molhere.name)
			
		#Below Section: Returns the recorded x-value snow line locations
		return {'radius':radvals, 'name':radnames}

		


			
			
#FUNCTION: calcmu NOT FINISHED
#PURPOSE: This function is meant to calculate the mean molecular weight of a given disk.
	def calcmu(self):
		#Below Section: Calculates mean molecular weight by considering contributions from all Mols in disk
		for molhere in self.mols:
			pass
			
		#Below Section: Returns calculated mu value
		return done
			