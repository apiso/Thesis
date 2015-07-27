#FILE: classMol.py
#PURPOSE: This file contains the class Mol, which models a particular molecular species.

class Mol(object):
#FUNCTION: __init__
#PURPOSE: This function initializes the current Mol with given parameters.
#INPUTS: energy = molecular bond breaking energy (K); mass = molecule mass (kg); xtot = xmolgr + xmolgas = total molecular fraction for molecule (over both grain and gas) of base nH; name = molecule chemical formula
	def __init__(self, xtot, mass, energy, name):
		#Automatic attributes
		self.xtot = xtot
		self.mass = mass
		self.energy = energy
		self.name = name
		self.elements = self.getElements(name)

		
#FUNCTION:
#PURPOSE: This function is meant to take the chemical formula of a molecule and separate it into element components with counted frequencies.
#EXAMPLE: 'H2O' would return the dictionary {'H':2, 'O':1}
#INPUTS: name = molecular species's chemical formula as string	
	def getElements(self, name):
		#Below Section: Records the elements in molecule
		elements = {}
		elhere = name[0]
		dighere = ''
		for a in range(0, len(name)):
			#Below handles if current letter in alphabet
			if name[a].isalpha():
				#If continued name of current element
				if name[a].islower():
					elhere = elhere + name[a]
										
				#Else if new element started
				elif name[a].isupper() and a > 0:
					#If a repeated element within this molecule
					if elhere in elements:
						if dighere == '':
							elements[elhere] = elements[elhere] + 1
						elif dighere != '':
							elements[elhere] = elements[elhere] + int(dighere)
					
					#If first time element added for molecule
					elif dighere == '':
						elements[elhere] = 1
					elif dighere != '':
						elements[elhere] = int(dighere)
					
					#Below resets elements
					elhere = name[a]
					dighere = ''
					
			#Else if number encountered
			elif name[a].isdigit():
				dighere = dighere + name[a]
					
			#Else if error encountered
			else:
				raise ValueError("Weird molecular name encountered at " + str(a) + "th index: " + name)

			#Below if end of name string reached
			if (a+1) >= len(name):
				#Records last element
				#If a repeated element within this molecule
				if elhere in elements:
					if dighere == '':
						elements[elhere] = elements[elhere] + 1
					elif dighere != '':
						elements[elhere] = elements[elhere] + int(dighere)
				
				#If first time element added for molecule
				elif dighere == '':
					elements[elhere] = 1
				elif dighere != '':
					elements[elhere] = int(dighere)
			
		#Below attributes the recorded elements
		return elements


#FUNCTION: __str__
#PURPOSE: This function is meant to specify what occurs when attempting to print an object Mol.
	def __str__(self):
		#Below generates strings of information to print
		strname = 'Molecule: ' + self.name + '\n'
		strels = 'Elements: ' + str(self.elements) + '\n'
		strmass = 'Mass (kg): {:.2e} \n'.format(self.mass)
		strenergy = 'Energy (K): {:.2f} \n'.format(self.energy)
		strxtot = 'Molecular Fraction of Total nH: {:.2e} \n'.format(self.xtot)
		
		#Below puts together string of information
		done = (strname + strels + strmass + strenergy + strxtot)
				
		#Below returns string to print
		return done

		