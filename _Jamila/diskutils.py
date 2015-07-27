#FILE: diskutils.py
#PURPOSE: This function is meant to provide certain base calculations and utilities for calculating various disk properties.
#USES: classMol, classDisk




######
###BACKGROUND
#Below Section: Imports necessary functions
import numpy as np
import matplotlib.pyplot as graph
import matplotlib as mpl
import astropy.constants as const

from classDisk import Disk
from classMol import Mol

#Below Section: Records necessary constants
pi = np.pi #Pi
kB = const.k_B.value #Boltzmann Constant
Gconst = const.G.value #Gravitational Constant
kBerg = const.k_B.cgs.value #Boltzmann Constant in cgs
au = const.au.value #Astronomical Unit (m)
mH = 1.661e-27 #Mass of H atom (kg)
msun = const.M_sun.value #Mass of Sun (kg)






######
###PLOTS
#FUNCTION: plotgradxel
#PURPOSE: This function is meant to plot the x_profile of an element within a given disk.  If two elements are given, will plot the ratio betwixt the two.
def plotgradxel(R, z=None, el1=None,
				el2=None, disk=None, cmap=graph.cm.rainbow,
				profTtype='basic', profnHtype='gaus-basic',
				xgr=1.0e-12, adsign=-1.0, design=1.0,
				rgr=((0.1e-6)*100), s=1, ratetot=0,
				snowlinemols=['H2O','CO2','CO','NH3','N2'],
				showsnowlines=False, ratioplot=False,
				root='PlotGrad-'):
			
	#Below Section: Makes sure disks and mols as correct inputs
	if not isinstance(disk, Disk) or not isinstance(el1, str):
		raise ValueError("Oh no!  Please make sure you input a Disk for the keyword 'disk' and the name of the element for the keyword 'el1'.")

	#Below sets up location matrix
	#Below sets up radius and height matrices
	Rmatr = np.resize(R, (len(z), len(R)))
	zmatr = np.resize(z, (len(R), len(z))).T
	
	#Below Section: Calculates for first element
	elstuff1 = calcnel(R=Rmatr, z=zmatr, el=el1, disk=disk, profTtype=profTtype, profnHtype=profnHtype, xgr=xgr, adsign=adsign, design=design, rgr=rgr, s=s, ratetot=ratetot, isgradient=True)
	xgasel1 = elstuff1['xgasel']
	xgrel1 = elstuff1['xgrel']
	
	#Below Section: Records element abundance values
	xgaselmatr = xgasel1
	xgrelmatr = xgrel1
	labelname = r'$x_'+el1+' / x_H$'
	savename = 'x'+el1
	if ratioplot: #If ratio is requested
		elstuff2 = calcnel(R=Rmatr, z=zmatr, el=el2, disk=disk, profTtype=profTtype, profnHtype=profnHtype, xgr=xgr, adsign=adsign, design=design, rgr=rgr, s=s, ratetot=ratetot, isgradient=True)
		xgasel2 = elstuff2['xgasel']
		xgrel2 = elstuff2['xgrel']		
		
		#Below records the elemental abundance values
		xgaselmatr = xgasel1/1.0/xgasel2
		xgrelmatr = xgrel1/1.0/xgrel2
		labelname = r'$x_'+el1+'$ / $x_'+el2+'$'	
		savename = 'x'+el1+'-'+el2
		
	#Below Section: Graphs the gaseous and grain contours
	fig = graph.figure()
	fig.suptitle(labelname, fontsize=26)
	minval = 0
	maxval = None
	cutoff = 10**-20
	if not ratioplot:
		maxval = np.max([np.max(xgaselmatr[~np.isnan(xgaselmatr)]), np.max(xgrelmatr[~np.isnan(xgrelmatr)])])
	else:
		maxval = 2
		
	#Below Section: Generates colorbar ticks and such
	numdenslines = 100
	cbarlines = np.linspace(minval, maxval, numdenslines, endpoint=True)
	cbarlinenames = None
	if not ratioplot: #Scientific notation for typical gradients
		cbarlinenames = ['{:.2e}'.format(here) for here in cbarlines]
	else:
		cbarlinenames = map(str, [round(here,1) for here in cbarlines])
	cbarlinenames[-1] = '>'+cbarlinenames[-1]
	#Below leaves only certain ticks
	for n in range(0, len(cbarlinenames)):
		if n == 0 or n == len(cbarlinenames)-1:
			continue
		if n % 10 != 0:
			cbarlinenames[n] = ''
	
	#Below Section: Graphs the contours
	#For gas
	#Below applies cutoff values
	xgaselmatr[xgaselmatr > maxval] = maxval
	xgaselmatr[xgaselmatr < cutoff] = minval
	#Graphs the gas portion
	figgas = graph.subplot(2,1,1)
	contoursgas = figgas.contourf(R, z, xgaselmatr, cbarlines, cmap=cmap, vmin=minval, vmax=maxval)
	#Below sets graph limits
	figgas.set_xlim([min(R), max(R)])
	figgas.set_ylim([min(z), max(z)])
		
	#For grain
	#Below applies cutoff values
	xgrelmatr[xgrelmatr > maxval] = maxval
	xgrelmatr[xgrelmatr < cutoff] = minval
	#Graphs the grain portion
	figgr = graph.subplot(2,1,2)
	contoursgr = figgr.contourf(R, z, xgrelmatr, cbarlines, cmap=cmap, vmin=minval, vmax=maxval)
	#Below sets graph limits
	figgr.set_xlim([min(R), max(R)])
	figgr.set_ylim([min(z), max(z)])

	#Below Section: Generates stuff for colorbar
	#Below generates the colorbar
	fig.subplots_adjust(right=.8)
	norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)
	cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
	cbar = graph.colorbar(contoursgas, ticks=cbarlines, cax=cbar_ax, norm=norm)
	
	#Below does colorbar labeling
	cbar.ax.set_yticklabels(cbarlinenames)
	cbar.set_label(labelname+r' [cm$^-$$^3$]', fontsize=14)
	cbar.ax.tick_params(labelsize=11)

	#Below Section: Adds snow lines if desired
	molstuff = disk.mols
	if showsnowlines:
		#Below grabs the x-values of given molecules for snow lines
		snowlinestuff = disk.getsnowlines(R=R, z=0.0, mollist=snowlinemols, profTtype=profTtype, profnHtype=profnHtype)
		snowlinexvals = snowlinestuff['radius']
		snowlinenames = snowlinestuff['name']
		for h in range(0, len(snowlinexvals)):
			#Points
			figgas.scatter(snowlinexvals[h], 0.0, s=30, marker='o', color = 'black', alpha=.4)
			figgr.scatter(snowlinexvals[h], 0.0, s=30, marker='o', color = 'black', alpha=.4)
			#Annotations
			figgas.annotate(snowlinenames[h], xy = (snowlinexvals[h], 0), xytext=(-10,15), textcoords='offset points', rotation=45, fontsize=14)
			figgr.annotate(snowlinenames[h], xy = (snowlinexvals[h], 0), xytext=(-10,15), textcoords='offset points', rotation=45, fontsize=14)
	
	#Below Section: Labels the graph
	figgas.annotate("Gaseous "+labelname, xy=(1, .2), xytext=(1, 55), textcoords='offset points', fontsize=16, bbox={'facecolor':'white', 'alpha':.6})
	figgr.annotate("Grain "+labelname, xy=(1, .2), xytext=(1, 55), textcoords='offset points', fontsize=16, bbox={'facecolor':'white', 'alpha':.6})
	figgr.set_xlabel('Radius (AU)', fontsize=16)
	figgas.set_ylabel("Height (AU)", fontsize=16)
	graph.savefig('Bin/'+root+savename+'.png')
	graph.close()

			
			

#FUNCTION: plotprofxel
#PURPOSE: This function is meant to plot the x_profile of an element within a given disk.  If two elements are given, will plot the ratio betwixt the two.
def plotprofxel(R, z=0.0, el1=None, el2=None, disk=None,
				profTtype='basic', profnHtype='gaus-basic',
				xgr=1.0e-12, adsign=-1.0, design=1.0,
				rgr=((0.1e-6)*100), s=1, ratetot=0,
				showsnowlines=False, ratioplot=None,
				snowlinemols=['H2O','CO2','CO','NH3','N2'],
				root='PlotProf-'):
	#Below Section: Makes sure disks and mols as correct inputs
	if not isinstance(disk, Disk) or not isinstance(el1, str):
		raise ValueError("Oh no!  Please make sure you input a Disk for the keyword 'disk' and the name of the element for the keyword 'el1'.")
	
	#Below Section: Calculates the elemental profile for the first given element
	elstuff1 = calcnel(R=R, z=z, el=el1, disk=disk,
					profTtype=profTtype, profnHtype=profnHtype,
					xgr=xgr, adsign=adsign, design=design,
					rgr=rgr, s=s, ratetot=ratetot)
	xgasel1 = elstuff1['xgasel']
	xgrel1 = elstuff1['xgrel']
	
	#Below Section: Graphs profiles for the first given element
	#Below adds snow lines if desired
	molstuff = disk.mols
	if showsnowlines:
		#Below grabs the x-values of given molecules for snow lines
		snowlinestuff = disk.getsnowlines(R=R, z=z, mollist=snowlinemols, profTtype=profTtype, profnHtype=profnHtype)
		snowlinexvals = snowlinestuff['radius']
		snowlinenames = snowlinestuff['name']
		for h in range(0, len(snowlinexvals)):
			graph.axvline(snowlinexvals[h], linestyle=':', linewidth=3, color = 'gray')
			graph.annotate(snowlinenames[h], xy = (snowlinexvals[h], 0), xytext=(0,30), textcoords='offset points', rotation=30, fontsize=16)
						
	#R vs. xel1
	graph.semilogx(R, xgasel1, color='red', linestyle='--', linewidth=3, label='Gaseous '+el1)
	graph.semilogx(R, xgrel1, color='purple', linestyle='-', linewidth=3, label='Grain '+el1)	
	
	#Below Section: Labels the plot
	graph.xlabel('Radius (AU)', fontsize=16)
	graph.ylabel(r'$x_'+el1+' / x_H$', fontsize=20)
	graph.legend(loc='upper right')
	#graph.suptitle('Disk: ' + disk.stringMol())
	#graph.title(r'$x_'+el1+'$', fontsize=16)
	graph.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	graph.ylim(ymin=0)
	graph.xlim(xmin=1)
	graph.savefig('Bin/'+root+'Rvsxel'+el1+'.png')
	graph.close()
	
	#Below Section: Calculates the elemental profile for the second given element, if given
	xel2 = None
	nel2 = None
	if el2 is not None:
		elstuff2 = calcnel(R=R, z=z, el=el2, disk=disk,
					profTtype=profTtype, profnHtype=profnHtype,
					xgr=xgr, adsign=adsign, design=design,
					rgr=rgr, s=s, ratetot=ratetot)
		xgasel2 = elstuff2['xgasel']
		xgrel2 = elstuff2['xgrel']
		
		#Below Section: Graphs profiles for the second given element, if given
		#Below Section: Graphs profiles for the first given element
		#Below adds snow lines if desired
		if showsnowlines:
			#Below grabs the x-values of given molecules for snow lines
			snowlinestuff = disk.getsnowlines(R=R, z=z, mollist=snowlinemols, profTtype=profTtype, profnHtype=profnHtype)
			snowlinexvals = snowlinestuff['radius']
			snowlinenames = snowlinestuff['name']
			for h in range(0, len(snowlinexvals)):
				graph.axvline(snowlinexvals[h], linestyle=':', linewidth=3, color = 'gray')
				graph.annotate(snowlinenames[h], xy = (snowlinexvals[h], 0), xytext=(0,30), textcoords='offset points', rotation=30, fontsize=16)

				
		#R vs. xel2
		graph.semilogx(R, xgasel2, color='red', linestyle='--', linewidth=3, label='Gaseous '+el2)
		graph.semilogx(R, xgrel2, color='purple', linestyle='-', linewidth=3, label='Grain '+el2)
		graph.xlabel('Radius (AU)', fontsize=16)
		graph.ylabel(r'$x_'+el2+' / x_H$', fontsize=20)
		graph.legend(loc='upper right')
		#graph.suptitle('Disk: ' + disk.stringMol())
		#graph.title(r'$x_'+el2+'$', fontsize=16)
		graph.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
		graph.ylim(ymin=0)
		graph.xlim(xmin=1)
		graph.savefig('Bin/'+root+'Rvsxel'+el2+'.png')
		graph.close()
	
		#Below Section: Calculates the ratio of the two elements
		#Below Section: Graphs profiles for the first given element
		#Below adds snow lines if desired
		if showsnowlines:
			#Below grabs the x-values of given molecules for snow lines
			snowlinestuff = disk.getsnowlines(R=R, z=z, mollist=snowlinemols, profTtype=profTtype, profnHtype=profnHtype)
			snowlinexvals = snowlinestuff['radius']
			snowlinenames = snowlinestuff['name']
			for h in range(0, len(snowlinexvals)):
				graph.axvline(snowlinexvals[h], linestyle=':', linewidth=3, color = 'gray')
				graph.annotate(snowlinenames[h], xy = (snowlinexvals[h], 0), xytext=(0,30), textcoords='offset points', rotation=30, fontsize=16)
	
		#R vs. xel1/xel2
		xgasratio = xgasel1/1.0/xgasel2
		xgrratio = xgrel1/1.0/xgrel2
		
		#Some naming conventions
		rationamesave = el1+'-'+el2
		rationamex = r'$x_'+el1+' / x_'+el2+'$'
		rationamen = r'$n_'+el1+' / n_'+el2+'$'
		rationame = el1+'/'+el2
		
		#The plotting
		graph.semilogx(R, xgasratio, color='red', linestyle='--', linewidth=3, label='Gaseous '+rationame)
		graph.semilogx(R, xgrratio, color='purple', linestyle='-', linewidth=3, label='Grain '+rationame)
		graph.xlabel('Radius (AU)', fontsize=16)
		if max(xgrratio[~np.isnan(xgrratio)]) > 1 or max(xgasratio[~np.isnan(xgasratio)]) > 1:
			graph.ylim([0,1.19])
		graph.ylabel(rationamex, fontsize=20)
		graph.legend(loc='upper right')
		#graph.suptitle('Disk: ' + disk.stringMol())
		#graph.title(rationamex)
		graph.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
		graph.ylim(ymin=0, ymax=1.89)
		graph.xlim(xmin=1)
		graph.savefig('Bin/'+root+'Rvsxel'+rationamesave+'.png')
		graph.close()
					
			

#FUNCTION: plotprofxelseries
#PURPOSE: This function is meant to plot the x_profile of an element within given disks for a series of plots.  If two elements are given each time, will plot the ratio betwixt the two.
def plotprofxelseries(Rlist, zlist=None, el1list=None,
				el2list=None, disklist=None,
				profTtype='basic', profnHtype='gaus-basic',
				xgr=1.0e-12, adsign=-1.0, design=1.0,
				rgr=((0.1e-6)*100), s=1, ratetot=0,
				showsnowlines=False, snowlinemols=None,
				notelist=['','','','',''],
				ratioplot=None,
				root='PlotProfSeries-'):
	#Below Section: Makes sure disks and mols as correct inputs
	if not isinstance(disklist[0], Disk) or not isinstance(el1list[0], str):
		raise ValueError("Oh no!  Please make sure you input a Disk for the keyword 'disk' and the name of the element for the keyword 'el1'.")
	
	#Below Section: Generates graph for each case given
	fig, grapharr = graph.subplots(len(Rlist), sharex=True, sharey='col')
	graph.xlabel('Radius (AU)', fontsize=16)
	for j in range(0, len(Rlist)):
		elstuff1here = calcnel(R=Rlist[j], z=zlist[j],
				el=el1list[j], disk=disklist[j],
				profTtype=profTtype, profnHtype=profnHtype,
				xgr=xgr, adsign=adsign, design=design,
				rgr=rgr, s=s, ratetot=ratetot)
		xgasel1 = elstuff1here['xgasel']
		xgrel1 = elstuff1here['xgrel']
	
		#Below Section: Graphs profiles for the first given element if requested
		if not ratioplot[j]:
			#Below adds snow lines if desired
			if showsnowlines:
				#Below grabs the x-values of given molecules for snow lines
				snowlinestuff = disklist[j].getsnowlines(R=Rlist[j], z=zlist[j], mollist=snowlinemols, profTtype=profTtype, profnHtype=profnHtype)
				snowlinexvals = snowlinestuff['radius']
				snowlinenames = snowlinestuff['name']
				for h in range(0, len(snowlinexvals)):
					grapharr[j].axvline(snowlinexvals[h], linestyle=':', linewidth=3, color = 'gray')
					grapharr[j].annotate(snowlinenames[h], xy = (snowlinexvals[h], 0), xytext=(0,30), textcoords='offset points', rotation=30, fontsize=20)

			#R vs. xel1
			grapharr[j].plot(Rlist[j], xgasel1, color='red', linestyle='--', linewidth=3, label='Gaseous '+el1list[j])
			grapharr[j].plot(Rlist[j], xgrel1, color='purple', linestyle='-', linewidth=3, label='Grain '+el1list[j])	
			grapharr[j].set_xscale('log')
		
			#Below Section: Labels the plot
			grapharr[j].set_ylabel(r'$x_'+el1list[j]+' / x_H$', fontsize=20)
			graph.legend(loc='upper right')
			graph.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
			grapharr[j].set_ylim(ymin=0)
			grapharr[j].set_xlim(xmin=1)
			grapharr[j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))					
			grapharr[j].locator_params(nbins=8, axis='y')
			graph.text(.5, .5, r'$x_'+el1list[j]+' '+notelist[j]+'$', fontsize=15)

		else:
			#Below Section: Calculates for the 2nd element passed in
			elstuff2here = calcnel(R=Rlist[j], z=zlist[j],
					el=el2list[j], disk=disklist[j],
					profTtype=profTtype, profnHtype=profnHtype,
					xgr=xgr, adsign=adsign, design=design,
					rgr=rgr, s=s, ratetot=ratetot)
			xgasel2 = elstuff2here['xgasel']
			xgrel2 = elstuff2here['xgrel']	
		
			#Below Section: Calculates the ratio of the two elements
			#Below adds snow lines if desired
			if showsnowlines:
				#Below grabs the x-values of given molecules for snow lines
				snowlinestuff = disklist[j].getsnowlines(R=Rlist[j], z=zlist[j], mollist=snowlinemols, profTtype=profTtype, profnHtype=profnHtype)
				snowlinexvals = snowlinestuff['radius']
				snowlinenames = snowlinestuff['name']
				for h in range(0, len(snowlinexvals)):
					grapharr[j].axvline(snowlinexvals[h], linestyle=':', linewidth=3, color = 'gray')
					grapharr[j].annotate(snowlinenames[h], xy = (snowlinexvals[h], 0), xytext=(0,30), textcoords='offset points', rotation=30, fontsize=16)
					
			#R vs. xel1/xel2
			xgasratio = xgasel1/1.0/xgasel2
			xgrratio = xgrel1/1.0/xgrel2
			
			#Some naming conventions
			rationamesave = el1list[j]+'-'+el2list[j]
			rationamex = r'$x_'+el1list[j]+' / x_'+el2list[j]+'$'
			rationamen = r'$n_'+el1list[j]+' / n_'+el2list[j]+'$'
			rationame = el1list[j]+'/'+el2list[j]
			
			#The plotting
			grapharr[j].plot(Rlist[j], xgasratio, color='red', linestyle='--', linewidth=3, label='Gaseous '+rationame)
			grapharr[j].plot(Rlist[j], xgrratio, color='purple', linestyle='-', linewidth=3, label='Grain '+rationame)
			grapharr[j].set_xscale('log')

			graph.xlabel('Radius (AU)', fontsize=16)
			if j == 0:
				grapharr[j].set_ylabel(rationamex, fontsize=20)
			grapharr[j].set_ylim(ymin=0)
			grapharr[j].set_xlim(xmin=1)
			grapharr[j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
			grapharr[j].locator_params(nbins=8, axis='y')
			#if max(xgrratio[~np.isnan(xgrratio)]) > 1 or max(xgasratio[~np.isnan(xgasratio)]) > 1:
			grapharr[j].set_ylim([0,1.19])
			if len(Rlist) == 2:
				grapharr[j].annotate(notelist[j], xy=(1, .2), xytext=(7.5, 120), textcoords='offset points', fontsize=16, bbox={'facecolor':'gray', 'alpha':.2})
			elif len(Rlist) == 3:
				grapharr[j].annotate(notelist[j], xy=(1, .2), xytext=(20, 60), textcoords='offset points', fontsize=16, bbox={'facecolor':'gray', 'alpha':.2})
				#xytext=(7.5, 70)
				

	#Below Section: Records the graph
	graph.legend(loc='upper right')
	#graph.suptitle('Disk: ' + disklist[j].stringMol())
	graph.subplots_adjust(hspace=0.05, wspace=None)
	graph.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	graph.savefig('Bin/'+root+'-TriplePlot.png')
	graph.show()
				
		
			
#FUNCTION: plotprofnel
#PURPOSE: This function is meant to plot the n_profile of an element within a given disk.  If two elements are given, will plot the ratio betwixt the two.
def plotprofnel(R, z=None, el1=None, el2=None, disk=None,
				profTtype='basic', profnHtype='basic',
				xgr=1.0e-12, adsign=-1.0, design=1.0,
				rgr=((0.1e-6)*100), s=1, ratetot=0,
				root='PlotProf-'):
	#Below Section: Makes sure disks and mols as correct inputs
	if not isinstance(disk, Disk) or not isinstance(el1, str):
		raise ValueError("Oh no!  Please make sure you input a Disk for the keyword 'disk' and the name of the element for the keyword 'el1'.")
	
	#Below Section: Calculates the elemental profile for the first given element
	elstuff1 = calcnel(R=R, z=z, el=el1, disk=disk,
					profTtype=profTtype, profnHtype=profnHtype,
					xgr=xgr, adsign=adsign, design=design,
					rgr=rgr, s=s, ratetot=ratetot)
	ngasel1 = elstuff1['ngasel']
	ngrel1 = elstuff1['ngrel']
	
	#Below Section: Graphs profiles for the first given element
	#R vs. nel1
	graph.semilogx(R, ngasel1, color='red', linestyle='--', linewidth=3, label='Gaseous '+el1)
	graph.semilogx(R, ngrel1, color='purple', linestyle='-', linewidth=3, label='Grain '+el1)
	graph.xlabel('Midplane Radius (AU)')
	graph.ylabel(r'$n_'+el1+' (cm^(-3))$')
	graph.legend(loc='best')
	graph.suptitle('Disk: ' + disk.stringMol())
	graph.title(r'Midplane Radius vs. $n_'+el1+'$')
	graph.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	graph.savefig('Bin/'+root+'Rvsnel'+el1+'.png')
	graph.close()
	
	#Below Section: Calculates the elemental profile for the second given element, if given
	xel2 = None
	nel2 = None
	if el2 is not None:
		elstuff2 = calcnel(R=R, z=z, el=el2, disk=disk,
					profTtype=profTtype, profnHtype=profnHtype,
					xgr=xgr, adsign=adsign, design=design,
					rgr=rgr, s=s, ratetot=ratetot)
		ngasel2 = elstuff2['ngasel']
		ngrel2 = elstuff2['ngrel']
		
		#Below Section: Graphs profiles for the second given element, if given
		#R vs. nel2
		graph.semilogx(R, ngasel2, color='red', linestyle='--', linewidth=3, label='Gaseous '+el2)
		graph.semilogx(R, ngrel2, color='purple', linestyle='-', linewidth=3, label='Grain '+el2)
		graph.xlabel('Midplane Radius (AU)')
		graph.ylabel(r'$n_'+el2+' (cm^(-3))$')
		graph.legend(loc='best')
		graph.suptitle('Disk: ' + disk.stringMol())
		graph.title(r'Midplane Radius vs. $n_'+el2+'$')
		graph.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
		graph.savefig('Bin/'+root+'Rvsnel'+el2+'.png')
		graph.close()
	
		#Below Section: Calculates the ratio of the two elements
		#R vs. xel1/xel2
		### 0/0 ---> nan ---> not plotted
		#Turns very small numbers into 0
		cutngasel1 = ngasel1.copy()
		cutngasel2 = ngasel2.copy()
		cutngrel1 = ngrel1.copy()
		cutngrel2 = ngrel2.copy()
		
		
		cutoff = 1e-10 ###Change to min el frac / 1e3
		cutngasel1[cutngasel1 < cutoff] = 0.0
		cutngasel2[cutngasel2 < cutoff] = 0.0
		cutngrel1[cutngrel1 < cutoff] = 0.0
		cutngrel2[cutngrel2 < cutoff] = 0.0
		
		
		ngasratio = cutngasel1/1.0/cutngasel2
		ngasratio = ngasratio[~np.isnan(ngasratio)]
		ngrratio = cutngrel1/1.0/cutngrel2
		ngrratio = ngrratio[~np.isnan(ngrratio)]
		
		Rngasratio = R[~np.isnan(ngasratio)]
		Rngrratio = R[~np.isnan(ngrratio)]
		
		#Some naming conventions
		rationamesave = el1+'-'+el2
		rationamex = r'$x_'+el1+' / x_'+el2+'$'
		rationamen = r'$n_'+el1+' / n_'+el2+'$'
		rationame = el1+'/'+el2
		
		#The plotting
		#R vs. nel1/nel2
		graph.semilogx(Rngasratio, ngasratio, color='red', linestyle='--', linewidth=3, label='Gaseous '+rationame)
		graph.semilogx(Rngrratio, ngrratio, color='purple', linestyle='-', linewidth=3, label='Grain '+rationame)
		graph.xlabel('Midplane Radius (AU)')
		graph.ylabel(rationamen)
		graph.legend(loc='best')
		graph.suptitle('Disk: ' + disk.stringMol())
		graph.title('Midplane Radius vs. '+rationamen)
		graph.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
		graph.savefig('Bin/'+root+'Rvsnel'+rationamesave+'.png')
		graph.close()
			
	
		

	
#FUNCTION: plotprofmol
#PURPOSE: This function is meant to plot the density and molecular fraction profiles of a given molecule.
def plotprofmol(R, z=None, disk=None, mol=None, root='PlotProf',
				profTtype='basic', profnHtype='basic',
				xgr=1.0e-12, adsign=-1.0, design=1.0,
				rgr=((0.1e-6)*100), s=1, ratetot=0):
	#Below Section: Makes sure disks and mols as correct inputs
	if not isinstance(disk, Disk) or not isinstance(mol, Mol):
		raise ValueError("Oh no!  Please make sure you input a Disk for the keyword 'disk' and a Mol for the keyword 'mol'.")

	#Below Section: Calculates the profiles
	#Profiles
	ntot = disk.profnH(R=R, z=z, type=profnHtype)

	#Below Section: Calculates ngas and ngr values at given radius
	molname = mol.name
	nmolvals = calcnmol(R=R, z=z, disk=disk, mol=mol,
				profTtype=profTtype,
				profnHtype=profnHtype, xgr=xgr,
				adsign=adsign, design=design, rgr=rgr,
				s=s, ratetot=ratetot)
	nmolgasvals = nmolvals[0]
	nmolgrvals = nmolvals[1]
	
	#Midplane Radius vs. nmolgas and nmolgr
	graph.semilogx(R, (nmolgasvals), color='red', linestyle='--', linewidth=3, label='Gaseous '+molname)
	graph.semilogx(R, (nmolgrvals), color='purple', linestyle='-', linewidth=3, label='Grain '+molname)
	graph.xlabel('Midplane Radius (AU)')
	graph.ylabel(r'$n_'+molname+' (cm^(-3))$')
	graph.legend(loc='best')
	graph.title(r'$Midplane Radius vs. n_'+molname+'$')
	graph.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	graph.savefig('Bin/'+root+'-Rvsnmol'+molname+'.png')
	graph.close()
		
	#Midplane Radius vs. xmolgas and xmolgr
	graph.semilogx(R, (nmolgasvals/1.0/ntot), color='red', linestyle='--', linewidth=3, label='Gaseous '+molname)
	graph.semilogx(R, (nmolgrvals/1.0/ntot), color='purple', linestyle='-', linewidth=3, label='Grain '+molname)
	#graph.plot(R, (Ntot/1.0/Ntot), color='gray', linewidth=3, linestyle=':', linewidth=3, label='Total Fraction of Base nH')
	graph.xlabel('Midplane Radius (AU)')
	graph.ylabel(r'$x_'+molname+' / x_H$')
	graph.legend(loc='best')
	graph.title(r'Midplane Radius vs. $x_'+molname+'$')
	graph.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	graph.savefig('Bin/'+root+'-Rvsxmol'+molname+'.png')
	graph.close()
	
	

#FUNCTION: plotprofbase
#PURPOSE: This function is meant to plot the temperature and density profiles for a given disk.
def plotprofbase(R, z=None, disk=None, mol=None,
				root='PlotProf-',
				profTtype='basic', profnHtype='basic'):
	#Below Section: Makes sure disks and mols as correct inputs
	if not isinstance(disk, Disk) or not isinstance(mol, Mol):
		raise ValueError("Oh no!  Please make sure you input a Disk for the keyword 'disk' and a Mol for the keyword 'mol'.")

	#Below Section: Calculates the profiles
	#Profiles
	temp = disk.profT(R=R, z=z, type=profTtype)
	ntot = disk.profnH(R=R, z=z, type=profnHtype)
	
	#Below Section: Plots relationships between profiles and varied parameters
	#Midplane Radius vs. Log10 nH
	graph.semilogy(R, ntot, color='black')
	graph.xlabel('Midplane Radius (AU)')
	graph.ylabel(r'$n_H (cm^(-3))$')
	graph.title(r'Midplane Radius vs. $n_H$')
	graph.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
	graph.savefig('Bin/'+root+'-RvsLog10nH.png')
	graph.close()
		
	#Midplane Radius vs. Log10 Temperature
	graph.semilogy(R, temp, color='blue')
	graph.xlabel('Midplane Radius (AU)')
	graph.ylabel('Temperature (K)')
	graph.title('Midplane Radius vs. Temperature')
	graph.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
	graph.savefig('Bin/'+root+'-RvsLog10T.png')
	graph.close()
			
		
	
#FUNCTION: plotprofgaus UNFINISHED
#PURPOSE: This function is meant to plot the contour of the dependences of temperature and density upon lateral and vertical radii, according to the Gaussian density profile and given temperature profile.
#INPUTS:
def plotprofgaus(R, z, disk=None, mol=None,
				root='PlotProf-', mu=1.0, mstar=1.0,
				denslogmin=7, denslogmax=14,
				templogmin=1, templogmax=2.5,
				numdenslines=15, numtemplines=12,
				tempcut=True,
				profTtype='cos', profnHtype='gaus-basic'):
	#Below Section: Makes sure disks and mols as correct inputs
	if not isinstance(disk, Disk):
		raise ValueError("Oh no!  Please make sure you input a Disk for the keyword 'disk'.")

	#Below Section: Calculates the profiles
	#Profiles
	templine = disk.profT(R=R, z=z, type=profTtype)
	tempmatr = np.resize(templine, (len(z), len(R)))	
	
	ntotmatr = None
	if profnHtype != 'gaus-cos':
		ntotmatr = np.zeros(shape=tempmatr.shape)
		for a in range(0, len(z)):
			ntotmatr[0:len(R)][a] = disk.profnH(R=R, z=z[a], type=profnHtype, mu=mu, mstar=mstar)
	else:
		ntotmatr = disk.profnH(R=R, z=z, type=profnHtype, mu=mu, mstar=mstar)
	
	#Below Section: Generates contour plots of the results
	#Temperature
	#Below generates colorbar ticks and such
	cbarlinesraw = np.linspace(templogmin, templogmax, numtemplines, endpoint=True)
	cbarlines = np.zeros(len(cbarlinesraw)+1)
	cbarlines[1:len(cbarlinesraw)+1] = cbarlinesraw
	cbarlinenames = map(str, [round(here,2) for here in cbarlines])
	#Below leaves only integer ticks
	for n in range(0, len(cbarlinenames)):
		if n == 0 or n == len(cbarlinenames)-1:
			continue
		if float(cbarlinenames[n])*2 != int(float(cbarlinenames[n])*2):
			cbarlinenames[n] = ''
	cbarlinenames[1] = '<'+cbarlinenames[1]
	cbarlinenames[-1] = '>'+cbarlinenames[-1]
	
	#Below graphs the contours
	#Temperature
	#Below cuts out temperature portions if requested
	if tempcut:
		tempmatr[np.log10(ntotmatr) >= denslogmax] = float('inf')
		tempmatr[np.log10(ntotmatr) <= denslogmin] = float('inf')
		
	contours = graph.contourf(R, z, np.log10(tempmatr), cbarlines, cmap=graph.cm.gist_heat)
	cbar = graph.colorbar(contours, ticks=cbarlines)
	cbar.ax.set_yticklabels(cbarlinenames)
	cbar.set_label("Log( Temperature [K] )", fontsize=16)
	#graph.title("Temperature Contours", fontsize=16)
	graph.xlabel('Radius (AU)', fontsize=16)
	graph.ylabel('Height (AU)', fontsize=16)
	graph.savefig('Bin/'+root+'GradTemp.png')
	graph.show()
	
	#Density
	#Below generates colorbar ticks and such
	cbarlinesraw = np.linspace(denslogmin, denslogmax, numdenslines, endpoint=True)
	cbarlines = np.zeros(len(cbarlinesraw)+1)
	cbarlines[1:len(cbarlinesraw)+1] = cbarlinesraw
	cbarlinenames = map(str, [round(here,2) for here in cbarlines])
	#Below leaves only integer ticks
	for n in range(0, len(cbarlinenames)):
		if n == 0 or n == len(cbarlinenames)-1:
			continue
		if float(cbarlinenames[n]) != int(float(cbarlinenames[n])):
			cbarlinenames[n] = ''
	cbarlinenames[1] = '<'+cbarlinenames[1]
	cbarlinenames[-1] = '>'+cbarlinenames[-1]
	
	#Below generates the contour graph
	contours = graph.contourf(R, z, np.log10(ntotmatr), cbarlines, cmap=graph.cm.bone)
	cbar = graph.colorbar(contours, ticks=cbarlines)
	cbar.ax.set_yticklabels(cbarlinenames)
	cbar.set_label(r"Log( n$_H$ [cm$^-$$^3$] )",fontsize=16)
	#graph.title(r"n$_H$ Contours", fontsize=16)
	graph.xlabel('Radius (AU)', fontsize=16)
	graph.ylabel('Height (AU)', fontsize=16)
	graph.savefig('Bin/'+root+'GradDens.png')
	graph.show()
	
	
	
#FUNCTION: plotprofcol
#PURPOSE: This function is meant to plot the column density of a protoplanetary disk.
#EQUATION: N = (2-y)(Md/(2*mH*pi (1 AU)^2)) (R/1 AU)^(-y)
#INPUTS:
#OUTPUTS: Column density in cm*-2
def plotprofcol(R, mstar=1, y=1, mdiskfactor=.004,
				taper=False, root='Try'):
	#Below Section: Calculates parts of the column density
	mdisk = mstar*mdiskfactor*msun
	Rratio = ((R/1.0)**(-1*y))
	if taper:
		taperinner = (-1) * ((R/1.0)**(2-y))
		Rratio = Rratio*np.exp(taperinner)
		
	norm = (2-y) * (1.0/(2*pi*((au*1.0)**2))) * mdisk/mH #For normalization

	#Below Section: Concatenates portions of density profile
	denscol = (norm*Rratio) / (100**2) #Converts to cm^-2
	
	#Below Section: Plots the column density
	graph.semilogy(R, denscol, color='brown')
	#graph.title(r'$n_H$ vs. Adsorption Rate')
	graph.ylabel(r'N$_H$ [cm$^-$$^2$]', fontsize=16)
	graph.xlabel('Radius (AU)', fontsize=16)
	graph.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
	graph.savefig('Bin/'+root+'nHcol.png')
	graph.close()
	
	
	
#FUNCTION: plotratesbase
#PURPOSE: This function is meant to plot the temperature and density nH profile of a given disk against the adsorption and desorption rates.
def plotratesbase(R, z=None, disk=None, mol=None,
					profTtype='basic', profnHtype='basic',
					xgr=1.0e-12, adsign=-1.0, design=1.0,
					rgr=((0.1e-6)*100), s=1, ratetot=0,
					root='PlotRates-'):
	#Below Section: Makes sure disks and mols as correct inputs
	if not isinstance(disk, Disk) or not isinstance(mol, Mol):
		raise ValueError("Oh no!  Please make sure you input a Disk for the keyword 'disk' and a Mol for the keyword 'mol'.")

	#Below Section: Calculates the rates of adsorption and desorption
	allrates = calcrates(R=R, z=z, disk=disk, mol=mol,
						profTtype=profTtype,
						profnHtype=profnHtype, xgr=xgr,
						rgr=rgr, s=s, adsign=adsign,
						ratetot=ratetot)				
	temp = allrates['T']
	ntot = allrates['ntot']
	ratead = allrates['ratead']
	ratede = allrates['ratede']
	
	#Below Section: Plots the rates against the density nH
	#nH vs. ratead
	graph.plot(ntot, ratead, color='black')
	graph.title(r'$n_H$ vs. Adsorption Rate')
	graph.xlabel(r'$n_H (cm^(-3))$')
	graph.ylabel(r'Adsorption Rate $(cm^(-3)*s^(-1))$')
	graph.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
	graph.savefig('Bin/'+root+'nHvsratead.png')
	graph.close()
		
	#nH vs. ratede
	graph.plot(ntot, ratede, color='black')
	graph.title(r'$n_H$ vs. Desorption Rate')
	graph.xlabel('n_H (cm^(-3))')
	graph.ylabel('Desorption Rate (cm^(-3)*s^(-1))')
	graph.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
	graph.savefig('Bin/'+root+'nHvsratede.png')
	graph.close()
		
	#Below Section: Plots the rates against the temperature
	#T vs. ratead
	graph.plot(temp, ratead, color='black')
	graph.title('Temperature vs. Adsorption Rate')
	graph.xlabel('Temperature (K)')
	graph.ylabel('Adsorption Rate (cm^(-3)*s^(-1))')
	graph.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
	graph.savefig('Bin/'+root+'Tvsratead.png')
	graph.close()
	
	#T vs. ratede
	graph.plot(temp, ratede, color='black')
	graph.title('Temperature vs. Desorption Rate')
	graph.xlabel('Temperature (K)')
	graph.ylabel('Desorption Rate (cm^(-3)*s^(-1))')
	graph.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
	graph.savefig('Bin/'+root+'Tvsratede.png')
	graph.close()
		
##



######
###METHODS
#FUNCTION: calcnel
#PURPOSE: This function is meant to calculate the number density of a given element within a given disk and return said number density and related quantities.
#OUTPUTS: Dictionary containing:
#	'nel': Element number density; 'xel': Element molecular fraction; 'mols': Names of molecules that factored into calculation; 'freqs': Frequencies for which element occurred in said molecules
def calcnel(R, el=None, z=None, disk=None,
				profTtype='basic', profnHtype='basic',
				xgr=1.0e-12, adsign=-1.0, design=1.0,
				rgr=((0.1e-6)*100), s=1, ratetot=0,
				isgradient=False):
	#Below Section: Makes sure disks and elements as correct inputs
	if not isinstance(disk, Disk) or not isinstance(el, str):
		raise ValueError("Oh no!  Please make sure you input a Disk for the keyword 'disk' and the name of the element for the keyword 'el'.")
		
	#Below Section: Calculates terms for determining rates
	#Profiles
	ntot = disk.profnH(R=R, z=z, type=profnHtype)

	#Below Section: Calculates nmol for each molecular species in the disk
	nmollist = [] #List of mol density profiles
	nmolnamelist = [] #List of names of mol density profiles
	freqlist = [] #List of frequency of el occurrence in related density profiles
	for molhere in disk.mols:
		if el in molhere.elements.keys():
			#Below calculates current mol density and frequency for which el occurs in mol
			nmolhere = calcnmol(R=R, z=z, disk=disk,
						mol=molhere, profTtype=profTtype,
						profnHtype=profnHtype, xgr=xgr,
						adsign=adsign, design=design,
						rgr=rgr, ratetot=ratetot)
			freqhere = molhere.elements[el]
			
			#Below records current density and frequency
			nmollist.append(nmolhere)
			nmolnamelist.append(molhere.name)
			freqlist.append(freqhere)
			
	#Below Section: Accumulates density by frequency
	nel = None
	if isgradient:
		nel = np.zeros(shape=(2, len(R), len(z)))
	else:
		nel = np.zeros(shape=(2, len(nmollist[0][0])))
	for a in range(0, len(nmollist)):
		nel[0] = nel[0] + (nmollist[a][0]*freqlist[a])
		nel[1] = nel[1] + (nmollist[a][1]*freqlist[a])
		
	#Below Section: Returns determined densities and frequencies
	return {'xgrel':(nel[1]/(1.0*ntot)), 'ngrel':nel[1], 'xgasel':(nel[0]/(1.0*ntot)), 'ngasel':nel[0], 'nmols':nmolnamelist, 'freqs':freqlist}
			
			
			
#FUNCTION: calcnmol
#PURPOSE: This function is meant to calculate the gaseous number density and grain number density of a molecular species from the combination of adsorption and desorption rates.
#ASSUMPTIONS: rgas << rgr; thermal velocity v; vgr << vgas, Ntot = xtot*ntot = nmolgas + nmolgr = N
#EQUATION: nmolgas = {(ntot*xtot)*v0*exp{-E/T}} / {(xgr*ntot*cross*s*v) + v0*exp{-E/T}}
#INPUTS: Midplane distance from central star 'R' (AU); vertical distance from midplain (z) (AU); xgr = grain molecular fraction; ntot = base comparative number density (1/cm^3); Disk = disk object (Disk); Mol = molecular species object (Mol); rgr = grain radius (cm); s = 'sticky' grain coefficient; ratetot = (adsorption rate + desorption rate)
#OUTPUTS: [nmolgas, nmolgr]
#NOTE: order1 uses desorption rate equation that switches between 0 and 1-order equations
def calcnmol(R, z=None, disk=None, mol=None,
				profTtype='basic', profnHtype='basic',
				xgr=1.0e-12, adsign=-1.0, design=1.0,
				rgr=((0.1e-6)*100), s=1, ratetot=0,
				order1=False):
	#Below Section: Makes sure disks and mols as correct inputs
	if not isinstance(disk, Disk) or not isinstance(mol, Mol):
		raise ValueError("Oh no!  Please make sure you input a Disk for the keyword 'disk' and a Mol for the keyword 'mol'.")
	
	#Below Section: Calculates terms for determining rates
	#Profiles
	temp = disk.profT(R=R, z=z, type=profTtype)
	ntot = disk.profnH(R=R, z=z, type=profnHtype)
	
	#Components
	v0 = calcv0(E=mol.energy, m=mol.mass)
	cross = pi*(rgr**2.0)
	vtherm = calcthermv(T=temp, m=mol.mass)
	exp = np.exp((-1.0*mol.energy)/temp)
	
	#Below Section: Calculates portions of equation
	botsuma = xgr*ntot*cross*s*vtherm
	botsumb = v0*exp	
	bot = (botsuma + botsumb)*1.0
	
	#Below Section: Calculates nmolgas and nmolgr
	#For nmolgas
	top = (ntot*mol.xtot)*v0*exp
	nmolgas = top/bot
	
	#For nmolgr
	nmolgr = (ntot*mol.xtot) - nmolgas
	
	############IN PROGRESS 7-25-15
	#Below Section: Puts desorption to 1st order if requested
	if order1:
		cutoff = 10.0**60 #Cutoff for 0-1 order threshold
		
		#Below are indices of order
#		ind0 = np.where(nmolgr >= cutoff)[0]
#		ind1 = np.where(nmolgr < cutoff)[0]
		
		#Below recalculates nmolgr and nmolgas with new values
		nmolgr[nmolgr >= cutoff] = cutoff
		nmolgas[nmolgr >= cutoff] = nmolgr[nmolgr >= cutoff]*botsumb[nmolgr >= cutoff]/1.0/(botsuma[nmolgr >= cutoff])
		"""
		cutoff = 10.0**6 #Cutoff for 0-1 order threshold
		tempdens = (ntot*mol.xtot).copy()
		tempdens[tempdens/1.0/cutoff >= 1.0] = cutoff

		#Below Section: Calculates portions of equation
		#botsuma = xgr*(tempdens/1.0/mol.xtot)*cross*s*vtherm
		#bot = (botsuma + botsumb)*1.0
	
		#Below calculates 0 and 1 order measurements
		top = tempdens*v0*exp

		#For nmolgas and nmolgr
		nmolgas = top/bot
		nmolgr = tempdens - nmolgas
		"""
	############
	
	#Below Section: Returns finished calculations
	return [nmolgas, nmolgr]

	
	
#FUNCTION: calcrates
#PURPOSE: This function is meant to calculate adsorption and desorption rates for some given molecular species.
#NOTES: Number densities are based on nH.  For example, ngr = grain number density = xgr*nH.
#ASSUMPTIONS: rgas << rgr; thermal velocity v; vgr << vgas
#INPUTS: Midplane distance from central star 'R' (AU); vertical distance from midplain (z) (AU); xgr = grain molecular fraction; xmolgas = gaseous molecular fraction; ntot = base comparative number density (1/cm^3); xtot = molecular fraction of base comparative number density; mmol = molecule species mass (kg); rgr = grain radius (cm); temp = temperature; E = oscillation break energy (K); s = 'sticky' grain coefficient
#OUTPUTS: {'ratead'=ratead, 'ratede'=ratede}, where ratead = adsorption rate, ratede = desorption rate
#NOTE: order1 uses desorption rate equation that switches between 0 and 1-order equations
def calcrates(R, z=None, disk=None, mol=None,
				profTtype='basic', profnHtype='basic',
				xgr=1.0e-12, rgr=(0.1e-6)*100, s=1,
				adsign=-1.0, design=0.0, ratetot=0,
				order1=False):
	#Below Section: Makes sure disks and mols as correct inputs
	if not isinstance(disk, Disk) or not isinstance(mol, Mol):
		raise ValueError("Oh no!  Please make sure you input a Disk for the keyword 'disk' and a Mol for the keyword 'mol'.")
	
	#Below Section: Calculates terms for determining rates
	#Profiles
	temp = disk.profT(R=R, z=z, type=profTtype)
	ntot = disk.profnH(R=R, z=z, type=profnHtype)

	#Components
	#Below calculates grain and gaseous number densities
	ngr = xgr*ntot
	nmol = calcnmol(R=R, z=z, disk=disk, mol=mol,
					profTtype=profTtype, adsign=adsign,
					profnHtype=profnHtype, design=design,
					xgr=xgr, rgr=rgr, ratetot=ratetot, s=s)
	nmolgas = nmol[0]
	nmolgr = nmol[1]
	
	#Below calculates cross-sectional area and velocity (thermal)
	v0 = calcv0(E=mol.energy, m=mol.mass)
	cross = pi*(rgr**2.0)
	vtherm = calcthermv(T=temp, m=mol.mass) #NOTE: Function converts to cm/s
	
	#Below Section: Calculates adsorption and desorption rates
	ratead = calcratead(ntot=ntot, xgr=xgr, nmolgas=nmolgas, cross=cross, vrel=vtherm, s=s)
	ratede = calcratede(nmolgr=nmolgr, T=temp, v0=v0, E=mol.energy)
	
	#Below Section: Returns calculated rates and other values
	return {'R':R, 'z':z, 'ratead':ratead, 'ratede':ratede, 'nmolgr':nmolgr, 'xmolgas':xmolgas, 'nmolgas':nmolgas, 'ntot':ntot, 'xtot':xtot, 'ngr':ngr, 'xgr':xgr, 'mmol':mmol, 'rgr':rgr, 'T':temp, 'v0':v0, 'E':E, 's':s, 'profTtype':profTtype, 'profnHtype':profnHtype}
	
##






######
###PROFILES
#FUNCTION: nmolprof
#PURPOSE: This function is meant to determine the profiles of nmolgas and nmolgr based upon given temperature profile and given density profile caculations.
#INPUTS: Midplane distance from central star 'R' (AU), vertical distance from midplain (z) (AU), Density reference point 'n0' (cm^-3*s^-1), Density power law index of power law 'yn', Temperature reference point 'T0' (K), Temperature power law index of power law 'yT'
#OUTPUTS: Gaseous number density set (gas and gr) nmol (cm^-3*s^-1)
def nmolprof(R, disk=None, mol=None, z=None,
				profTtype='basic', profnHtype='basic',
				xgr=1.0e-12, rgr=(0.1e-6)*100, s=1,
				adsign=-1.0, design=1.0, ratetot=0):
	#Below Section: Makes sure disks and mols as correct inputs
	if not isinstance(disk, Disk) or not isinstance(mol, Mol):
		raise ValueError("Oh no!  Please make sure you input a Disk for the keyword 'disk' and a Mol for the keyword 'mol'.")

	#Below Section: Calculates temperature and density profile values at given radius
	temp = disk.profT(R=R, z=z, type=profTtype)
	ntot = disk.profnH(R=R, z=z, type=profnHtype)

	#Below Section: Calculates nmol value at given locations
	nmol = calcnmol(R=R, z=z, disk=disk, mol=mol,
				profTtype=profTtype, profnHtype=profnHtype,
				xgr=xgr, adsign=adsign, design=design,
				rgr=rgr, s=s, ratetot=ratetot)
	
	#Below Section: Returns the calculated ngas value stuff
	return nmol
	
	

#FUNCTION: densprofgaus
#PURPOSE: This function is meant to provide a density profile using a Gaussian solution as provided in the Andrews "Observations of Protoplanetary..." paper.
#EQUATION: dens = (N/(H*sqrt(2*pi))) * exp{(-1/2)*(z/H)^2}
#INPUTS: mu = mean molecular weight; mstar = central star mass (M_Sun), N = column density
#OUTPUTS: dens (cm^-3)
def densprofgaus(R, z, T, mu=1.0, mstar=1.0, y=1,
					mdiskfactor=.004, taper=False):
	#Below Section: Calculates portions of equation
	mdisk = mstar*mdiskfactor*msun
	H = calcscaleH(R=(R*au), mu=mu, mstar=(mstar*msun), T=T)
	Rratio = ((R/1.0)**(-1*y))
	if taper:
		taperinner = (-1) * ((R/1.0)**(2-y))
		Rratio = Rratio*np.exp(taperinner)
	
	constpart = Rratio/1.0/(H*(np.sqrt(2*pi)))
	exppart = np.exp((-1.0/2.0)*(((z*au)/1.0/H)**2))
		
	norm = (2-y) * (1.0/(2*pi*((au*1.0)**2))) * mdisk/mH #For normalization

	#Below Section: Returns calculated density profile
	return (norm*constpart*exppart) / (100**3) #Converts to cm^-3
		
	

#FUNCTION: densprofbasic
#PURPOSE: This function is meant to provide a basic density profile based upon a given Midplane radius in the disk.
#ASSUMPTIONS: Assumes density as constant along vertical direction z in disk
#EQUATION: n = n0*(R/1.0(AU))**yn
#NOTES: Built specifically for nH
#INPUTS: Midplane distance from central star 'R' (AU); vertical distance from midplain (z) (AU); Reference point 'n0' (cm^-3*s^-1), power law index of power law 'yn'
#OUTPUTS: Number density n (cm^-3*s^-1)
def densprofbasic(R, z=None, n0=10.0**10, yn=(-1)):
	#Below Section: Calculates portions of profile
	part = R/1.0
	done = n0*(part**yn)

	#Below Section: Returns the calculated profile
	return done
	

	
#FUNCTION: tempprofcos
#PURPOSE: This function is meant to provide a 2D temperature profile based upon a given radius and height of the disk.
#EQUATION: T = (TA + (Tmid - TA)cos(pi*z/(2*4*H))^(2d))
#	Where TA = T0((R/1 AU)^-q)
#INPUTS: R = radius (AU), z = height (AU)
#OUTPUTS: Temperature T (K)
def tempprofcos(Tmid, R, z, d=2,
				mu=1.0, mstar=1.0, T0=343.269, q0=0.54):
	#Case of one z-value
	if isinstance(z, float) or isinstance(z, int):
		#Below Section: Puts together parts of equation
		TA = T0*((R/1.0)**(-1*q0))
		H = calcscaleH(R=R*au, mu=mu, T=Tmid, mstar=mstar*msun) #In (m)

		#Below Section: Concatenates more parts
		cosinner = pi*(z*au) / (2.0*(4*H)) #Unitless
		cospart = np.cos(cosinner)**(2*d) #Unitless
		tempdone = TA + ((Tmid - TA)*cospart)
		#Below Section: Returns finished calculation
		return tempdone	
	
	#Else, case of z-array
	#Below Section: Puts together R-z matrix for locations
	matr = np.meshgrid(R, z)[0]
	for zind in range(0, len(z)):
		#Below Section: Puts together parts of equation
		TA = T0*((R/1.0)**(-1*q0))
		H = calcscaleH(R=R*au, mu=mu, T=Tmid, mstar=mstar*msun) #In (m)

		#Below Section: Concatenates more parts
		cosinner = pi*(z[zind]*au) / (2.0*(4*H)) #Unitless
		cospart = np.cos(cosinner)**(2*d) #Unitless
		matr[zind, 0:len(R)] = TA + ((Tmid - TA)*cospart)
			
	#Below Section: Returns finished calculation	
	return matr
	
	
	
#FUNCTION: tempprofbasic
#PURPOSE: This function is meant to provide a basic temperature profile based upon a given Midplane in the disk.
#ASSUMPTIONS: Assumes temperature as constant along vertical direction z in disk
#EQUATION: T = T0*(R/1.0(AU))**yT
#INPUTS: Midplane distance from central star 'R' (AU); vertical distance from midplain (z) (AU); Reference point 'T0' (K), power law index of power law 'yT'
#OUTPUTS: Temperature T (K)
def tempprofbasic(R, z=None, T0=200.0, yT=(-0.62)):
	#Below Section: Calculates portions of profile
	part = R/1.0
	done = T0*(part**yT)
	
	#Below Section: Returns the calculated profile
	return done

##






######
###UTILITIES
#FUNCTION: calcscaleH
#PURPOSE: This function is meant to calculate the scale height of a disk.
#EQUATION: H = [ (k*T/(u*mH))*(r^3/(G*Ms)) ]^(1/2)
#INPUTS: R = lateral distance (m), T = temp (K), mu = molecular fraction, mstar = central star mass (kg)...
#OUTPUTS: H (m)
def calcscaleH(R, mu, T, mstar):
	#Below Section: Calculates portions of equation
	firstpart = (kB*T)/1.0/(mu*mH)
	secondpart = (R**3)/(Gconst*mstar)
	done = (firstpart*secondpart)**(1.0/2.0)
	
	#Below Section: Returns calculated scale height
	return done



#FUNCTION: calcratead
#PURPOSE: This function is meant to calculate the rate of adsorption.
#EQUATION: ratead = (ngr)*(ngas)*(cross)*(vrel)*(s)
#INPUTS: ngr = grain number density (1/cm^3); nmolgas = gaseous number density (1/cm^3); cross = cross-sectional area (cm^2); vrel = absolute value of relative velocity (cm/s); s = 'sticky' grain coefficient; ntot = base comparative number density (like nH) (1/cm^3); xgr = grain fraction of ntot; ngr = grain number density (1/cm^3) = xgr*ntot; 
#OUTPUTS: ratead = rate of adsorption
def calcratead(ntot, xgr, nmolgas, cross, vrel, s):
	#Below Section: Calculates adsorption rate
	ngr = ntot*xgr
	ratead = ngr*nmolgas*cross*s*vrel
	
	#Below Section: Returns finished calculation
	return ratead
	

#FUNCTION: calcratede
#PURPOSE: This function is meant to calculate the rate of desorption.
#EQUATION: ratede = (ngr)*(v0)*exp{-E/T}
#INPUTS: nmolgr = grain number density of molecule; v0 = oscillation frequency of molecular species bond to grain (1/s); E = energy to break molecular species bond to grain (K); T = temperature (K)
#OUTPUTS: ratede = rate of desorption
def calcratede(nmolgr, T, v0, E):
	#Below Section: Calculates portions of desorption rate
	exp = np.exp((-1.0)*E/(1.0*T))
	part = nmolgr*v0
	ratede = exp*part
	
	#Below Section: Returns finished calculation
	return ratede
	

#FUNCTION: calcthermv
#PURPOSE: This function is meant to calculate the thermal velocity of an item given the item's temperature and mass.
#EQUATION: v = sqrt{(8kT)/(pi*m)}
#	Where k = Boltzmann constant
#INPUT: Temperature T (K), Mass m (kg)
#OUTPUT: Thermal Velocity v (cm/s)
def calcthermv(T, m):
	#Below calculates top and bottom of equation
	top = 8.0*kB*T
	bot = pi*m*1.0

	#Below puts parts together and returns calculation
	done = np.sqrt(top/bot)
	return done*100 #Converts to cm/s
	

#FUNCTION: calcv0
#PURPOSE: This function is meant to calculate the oscillation frequency for a given molecular species.
#EQUATION: v0 = sqrt{ (2*kBerg*Nsites*E) / (pi^2 * m) }
#INPUT: Number density of molecular sites Nsites (cm^-2); Molecular mass m (kg); Oscillation break energy E (K)
#OUTPUT: Oscillation frequency v0 (1/s)
def calcv0(E, m, Nsites=1.0e15):
	#Below Section: Calculates parts of equation
	top = 2.0*kBerg*Nsites*E
	bot = (pi**2)*(m*1000.0)
	
	#Below Section: Puts together and returns equation
	done = np.sqrt(top/(1.0*bot))
	return done


#FUNCTION: calcquad
#PURPOSE: This function is meant to calculate the quadratic roots for the equation ax^2 + bx + c, using the coefficients of a, b, and c.
#EQUATION: (roots) = [-b +/- sqrt{b^2 - 4ac}]/2a
#INPUT: Coefficients a, b, c
#OUTPUT: Proposed values of x
def calcquad(a, b, c):
	#Below calculates part of the formula
	inner = np.sqrt(b**2.0 - 4*a*c)
	change = -1.0*b
	bot = 2.0*a
	
	#Below calculates the two different roots
	rootpos = (change + inner)/bot
	rootneg = (change - inner)/bot
	
	#Below Section: Returns the calculated roots
	return [rootpos, rootneg]


	
#FUNCTIONS: findnearest
#PURPOSE: Finds nearest value to something in an array.
def findnearest(arr, val):
	ind = (np.abs(arr-val)).argmin()
	return {'index':ind, 'value':arr[ind]}
	
##