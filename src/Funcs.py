###########################
#
# PyFriends Function Bank
# Trystan Lambert 
#
###########################

### Import ### 
import numpy as np 
import warnings
from scipy.integrate import quad
from scipy.integrate import cumtrapz
from astropy import units as u
from astropy.coordinates import SkyCoord
import datetime    
import time
warnings.filterwarnings('error')  #catch the cos(dec) error that sometimes we get an invoke long astropy method in these instances.
from tqdm import tqdm

########################################
########################################
# 									   #
#			GENERAL FUNCTIONS 		   #
#									   #
########################################
########################################


###################################################################################
# Functions to read in the input parameters as set by the user and to calculate any 
# constants that are needed throughout running the script. (These include once off
# calculations which can be called as variables instead of calculating them many 
# times troughout the program)
#
# input: name of text file with input parameters

# output: little h, Magnitude Limit of Survey, fiducial velocity, begining velcoity, 
#         ending velocity, projected limit, velocity limit, number of runs to be done, 
# 		  initial d0 value, final d0 value, v0 constant value, alpha (shecter), 
# 		  M_star (shecter), Phi star(shecter), H0, M_lim (Huchra & Geller, 1982), lum_const
###################################################################################


def read_in_params(infile):
	h,O_lam,O_m,O_k,MagLim,vf,red_start,redlim,Proj_Limit,Vel_Limit,runs,d0_i,d0_f,v0,cutoff,alpha,M_star,Phi_star=np.loadtxt('Params.txt',usecols=(1))
	runs = int(runs)
	Phi_star=Phi_star*(h**3)
	H0 = 100*h
	M_lim = MagLim-25-5*np.log10(vf/H0)
	lum_const=0.4*np.log(10)*Phi_star
	Dh=3000*(1./h)
	return h,O_lam,O_m,O_k,Dh,MagLim,vf,red_start,redlim,Proj_Limit,Vel_Limit,runs,d0_i,d0_f,v0,cutoff,alpha,M_star,Phi_star,H0,M_lim,lum_const


def calculate_params():
	integral1, err1 = quad(LuminosityFunction,-300,M_lim)
	return integral1


###################################################################################
# Function used to average over a "circular" data set. I.e. if we were to average 
# over lines of longitude the average of 359 and 1 would be 360/2 = 180 which is 
# on the complete wrong side of the sky. The answer should be 0. This function 
# corrrects that.
#
# input: one numpy array.

# output: A single float. 
###################################################################################
def WrapMean(array):  
	if (np.max(array)-np.min(array)>=180) and len(np.where((array>90) & (array<270))[0])==0:
		left=[]
		right=[]
		for k in range(len(array)):
			if array[k]<180:
				right.append(array[k])
			else:
				left.append(array[k])
		left_avg=np.mean(left)-360
		right_avg=np.mean(right)
		avg=np.mean([left_avg,right_avg])
		if avg<0:
			avg+=360
	else:
		avg=np.mean(array)
	return avg


###################################################################################
# Average Magnitude between two galaxies with some average velocity between them 
# (see Huchra & Geller, (1982) & Crook et al., (2007)).
#
# input: The average velocity of the two galaxies as a single value and the Magnitude
# 		 limit of the survey (MagLim should be read in from Params.txt).
#
# output: Single float of the average M12 value as defined in Huchra & Geller, (1982).
###################################################################################

def M12(v_avg):
	return MagLim-25-5*np.log10(v_avg/H0)


###################################################################################
# Shecter Luminosity Function as a function of Magnitude. This is parametized by 
# M_star, alpha, and phi star (see Crook et al., (2007)). These parameters should be 
# read in as input in the Params.txt file
# 
# input: The array of Magnitude (M) values.
# 
# output: An output array representing the Y values in the Shecter plot.
###################################################################################

def LuminosityFunction(M):  #Used in all the integrals 
	t2=10**(0.4*(alpha+1)*(M_star-M))
	t3=np.exp(-10**(0.4*(M_star-M)))
	return lum_const*t2*t3


###################################################################################
# Function which calculates the angular separation of two sources on sky in degrees
#
# input: ra of the first source, ra of the second source, dec of the first source
#        dec of the second source. ra's and dec's can be either floating points or 
#        arrays so long as they are consistent. i.e. ra1 float, dec1 float, ra2 array
#        dec2 array etc. 
#
# output: single floating point value representing the angular separation
###################################################################################

def angsep(ra1,ra2,dec1,dec2):
	try:
		faq=(np.pi/180)
		ra_1,ra_2,dec_1,dec_2=faq*ra1,faq*ra2,faq*dec1,faq*dec2
		cosl=np.cos((np.pi/2)-dec_2)*np.cos((np.pi/2)-dec_1)+np.sin((np.pi/2)-dec_2)*np.sin((np.pi/2)-dec_1)*np.cos(ra_2-ra_1)
		val=(1./faq)*np.arccos(cosl)
	except RuntimeWarning:
		c1=SkyCoord(ra=ra1*u.degree,dec=dec1*u.degree,frame='icrs')
		c2=SkyCoord(ra=ra2*u.degree,dec=dec2*u.degree,frame='icrs')
		sep=c1.separation(c2)
		val=sep.value

	return val

###################################################################################
# Function to calculate the projected onsky distance of an array of points against
# a single value 
#
# input: first ra, second ra, first dec, second dec, first v, second v. Input can be 
#        two floting points or a floating point and an array
#
# output: depending on input either a single floating point or an array of projected 
#         on-sky distances
###################################################################################

def Projected_OnSky_Distance(ra1,ra2,dec1,dec2,v1,v2):
	separations = angsep(ra1,ra2,dec1,dec2)
	theta = (np.pi/180)*(separations/2)
	mean_v = (v1 + v2)/2
	projected_distance = np.sin(theta)*(mean_v/H0)
	return projected_distance



########################################
########################################
# 									   #
#	    FOF algorithm functions		   #
#									   #
########################################
########################################


###################################################################################
# Base function for finding friends. Takes an index which points to the position of 
# a galaxy in some array/arrays and finds all the "friends", which are galaxy which 
# meet the criteira of being friends (defined in Huchra & Geller et al., (1982))
#
# input: index of the galaxy, velocity array, ra array, dec array, v0 constant,
#        d0 constant.
###################################################################################

def FindFriends(galaxy_index,v,ra,dec,v0,d0): #add checked to the end if need be
	
	v_cut=np.where((v>=v[galaxy_index]-v0) & (v<=v[galaxy_index]+v0))[0]  				# Find all the galaxies in the v array within +- v0

	ra_friends=ra[v_cut]   																# Narrow search in all arrays to just these values
	dec_friends=dec[v_cut]
	v_friends=v[v_cut]

	separations=angsep(ra_friends,ra[galaxy_index],dec_friends,dec[galaxy_index]) 		# work out the separations in projected distance (THIS NEEDS TO BE ITS OWN FUNCTION)
	#theta=(np.pi/180)*(separations/2)
	v_average=(v[galaxy_index]+v_friends)/2
	#D12=(np.sin(theta)*(v_average/H0))
	D12 = Projected_OnSky_Distance(ra_friends,ra[galaxy_index],dec_friends,dec[galaxy_index],v_friends,v[galaxy_index])

	M12s=M12(v_average)  																# Calculate the M12 values for all of the values 
	M12sort=np.sort(M12s)
	arg=M12s.argsort()
	rev=arg.argsort()

	M12sort=np.append(np.array([-32]),M12sort) #add first step
	yts=LuminosityFunction(M12sort)
	integrals=cumtrapz(yts,M12sort)
	integrals=integrals[rev] #return to correct order

	D_L=d0*((integrals/integral1)**(-1./3))
	
	checklimit=np.where(D_L>projected_limit)[0]				#making sure that the linking length can never be larger than the projected limits (at high cz)
	if len(checklimit)<>0:
		D_L[checklimit]=projected_limit-0.1

	pos_distances=D_L-D12
	pos=np.where(pos_distances>=0)[0]  
	return v_cut[pos]


###################################################################################
# 
###################################################################################

def FindGroupQuick(galaxy_index,v,ra,dec,v0,d0,group_vlim,lim_rad):
	friends_after=FindFriends(galaxy_index,v,ra,dec,v0,d0)
	friends_before=[]
	new=np.setdiff1d(friends_after,np.array([galaxy_index]))
	iterations=0	
	while list(friends_after) <> list(friends_before) and iterations<20:
		iterations+=1
		friends_before=friends_after

		#update the group center
		group_ra=WrapMean(ra[friends_before])
		group_dec=np.mean(dec[friends_before])
		group_v=np.mean(v[friends_before])

		V_cut=np.where((np.abs(v[friends_before]-group_v)<=group_vlim))[0]  #returns which array indicies of the current group satisy this condition

		RA_friends=ra[friends_before][V_cut]      #excludes all the galaxies which fall out of the velocity limits 
		DEC_friends=dec[friends_before][V_cut]
		V_friends=v[friends_before][V_cut]

		DD12 = Projected_OnSky_Distance(RA_friends,group_ra,DEC_friends,group_dec,V_friends,group_v)

		Pos=np.where(DD12<=lim_rad)[0]
		friends_before=friends_before[V_cut][Pos]
		new=np.intersect1d(friends_before,new)
		variable=[]
		for i in range(len(new)):
			variable+=list(FindFriends(new[i],v,ra,dec,v0,d0))
		friends_after=np.unique(np.append(friends_before,variable)).astype(int)
		new=np.setdiff1d(friends_after,friends_before)

		#print '\t', len(friends_before), len(new), len(friends_after)
	return friends_after



###################################################################################
# 
###################################################################################

def run_fof(Ra,Dec,v,v0,d0,vlim,dlim):

	checked=np.zeros(len(Ra))
	Groups=[]
	#tic=datetime.datetime.now() 
	local_cut=checked
	local_cut=np.where(checked==0)[0]   
	while len(local_cut)>1:   
		local_cut=np.where(checked==0)[0]
		local_p=np.random.randint(0,len(local_cut))
		#print len(local_cut)
		local_ra=Ra[local_cut]
		local_dec=Dec[local_cut]
		local_v=v[local_cut]
		local_group=FindGroupQuick(int(local_p),local_v,local_ra,local_dec,v0,d0,vlim,dlim)
		group=local_cut[local_group]
		Groups.append(group)       
		checked[group]=1
		#local_cut=np.where(checked==0)[0]
	#toc=datetime.datetime.now()
	#print toc-tic

	groups=[]
	notgroups=[]
	for i in range(len(Groups)):
		if len(Groups[i])>2:
			groups.append(Groups[i])
		else:
			notgroups.append(Groups[i])

	if len(notgroups)>1:    #need at least one array to concatenate or else it wont work
		notgroups=np.concatenate(notgroups)

	return groups,notgroups


###################################################################################
# 
###################################################################################

def FoF(Ra,Dec,v,Vel_Limit,projected_limit,output_file):
	dd0=d0_f-d0_i
	it=dd0/runs
	f=open(output_file,'w')
	f.close()
	for i in range(runs):
		f=open(output_file,'a')
		print i#'\t\t RUN NUMBER ', i 

		d0=d0_i+it*i
		print '\t\t d0 = ', d0
		t=run_fof(Ra,Dec,v,v0,d0,Vel_Limit,projected_limit)
		
		for trial in range(len(t[0])):
			for incident in range(len(t[0][trial])):
				f.write(str(t[0][trial][incident])+' ')
			f.write('\n')
		f.close()


##### Reading in Constants #####

h,Om_e,Om_m,Om_k,Dh,MagLim,vf,red_start,redlim,projected_limit,Vel_Limit,runs,d0_i,d0_f,v0,cutoff,alpha,M_star,Phi_star,H0,M_lim,lum_const = read_in_params('Params.txt')
integral1 = calculate_params()


#cosmology section. Defining these functions to get heliocentric velocities into CMB velocities. 
def VCMB(glon,glat,v):
	Lapex=264.14
	Bapex=48.26
	Vapex=371.0

	dL=glon-Lapex
	T1=np.sin(glat*(np.pi/180))*np.sin(Bapex*(np.pi/180))
	T2=np.cos(glat*(np.pi/180))*np.cos(Bapex*(np.pi/180))*np.cos(dL*(np.pi/180))

	return v+(Vapex*(T1+T2))

def E(z):
	term1=Om_m*((1+z)**3)
	term2=Om_k*((1+z)**2)
	term3=Om_e
	return np.sqrt(term1+term2+term3)

def integrand(z):
	k=E(z)
	return 1./k

def CoMoving(cz):
	cm=[]
	for i in range(len(cz)):
		integral,err=quad(integrand,0,float(cz[i]/300000.))
		cm.append(Dh*integral)
	return np.array(cm)

def Get_Dist(glon,glat,v_helio):
	cmb_velocity=VCMB(glon,glat,v_helio)
	cmb_redshift=cmb_velocity/300000. 
	cmb_comoving_distance=CoMoving(cmb_velocity)
	return cmb_velocity,cmb_redshift,cmb_comoving_distance

def Convert_to_XYZ(glon,glat,v_helio):
	cmb_v,cmb_z,cmb_d,=Get_Dist(glon,glat,v_helio)
	c=SkyCoord(l=glon*u.degree,b=glat*u.degree,distance=cmb_d*u.Mpc,frame='galactic')
	val_x=c.cartesian.x.value
	val_y=c.cartesian.y.value
	val_z=c.cartesian.z.value
	return cmb_v,cmb_z,cmb_d,val_x,val_y,val_z