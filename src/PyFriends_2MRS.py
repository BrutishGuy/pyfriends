
#################################
# Friends-of-Friends implementation as used in Lambert et. al., (2020)
#################################


import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import cumtrapz
import datetime    
import time 
import os
from astropy import units as u
from astropy.coordinates import SkyCoord
import warnings
from tqdm import tqdm
import GM 
import Py2Friends as Funcs
import IAPUC


#using py2Fridend on the completed 2MRS Redshift Survey

warnings.filterwarnings('error')  #catch the cos(dec) error that sometimes we get an invoke long astropy method in these instances.

h,Om_e,Om_m,Om_k,Dh,MagLim,vf,red_start,redlim,projected_limit,Vel_Limit,runs,d0_i,d0_f,v0i,v0f,cutoff,alpha,M_star,Phi_star,H0,M_lim,lum_const= Funcs.read_in_params('../config.txt')
integral1 = Funcs.calculate_params()

######################################################## This is just getting the Ra,Dec,v (plus ID's) ###################################################
infile="../data/2mrs_1175_done.dat"
Ra, Dec, l, b, v, K = np.loadtxt(infile, usecols=(1,2,3,4,24,5),unpack=True)  #readining in numerical data
MASXJIDs,other_names=np.loadtxt(infile,usecols=(0,28),dtype=str,unpack=True) #reading in float data

galaxyid=np.arange(0,len(Ra),1)
cut=np.where((v>red_start)&(v<redlim))
Ra,Dec,l,b,v,K,galaxyid,MASXJIDs,other_names=Ra[cut],Dec[cut],l[cut],b[cut],v[cut],K[cut],galaxyid[cut],MASXJIDs[cut],other_names[cut]
###########################################################################################################################################################

#### This is the entire FoF algorithm being run in a single line ####
start_time = datetime.datetime.now()
groups,edges,weighting,weighting_normed,sub_groupings,notgroups = Funcs.FoF_full(Ra,Dec,v)
end_time = datetime.datetime.now()
print 'Program Time:', end_time-start_time
print
######################################################################

####################################################################################################################
# At this point the algorithm is done running and we have an array of arrays with the groups and not groups
#  This is formating into the various different formats that we want. 
####################################################################################################################

########################################################################################
# Writing the edges meta data, using the same cosmology as everything else
########################################################################################
print 'Writing to File'
prefix_string='2MRS'


edge_1,edge_2,weight=np.zeros(len(edges)),np.zeros(len(edges)),np.zeros(len(edges)) #split the edges tuples triplates into 3 different arrays
for i in range(len(edges)):
	edge_1[i]=edges[i][0]
	edge_2[i]=edges[i][1]
	weight[i]=edges[i][2]

weight=weight.astype(str)

l1=l[edge_1.astype(int)]
l2=l[edge_2.astype(int)]
b1=b[edge_1.astype(int)]
b2=b[edge_2.astype(int)]
vh1=v[edge_1.astype(int)]
vh2=v[edge_2.astype(int)]

vcmb1,_,dist1,x1,y1,z1 = Funcs.Convert_to_XYZ(l1,b1,vh1)
vcmb2,_,dist2,x2,y2,z2 = Funcs.Convert_to_XYZ(l2,b2,vh2)

#write as an ascii format
labels=['X1','Y1','Z1','X2','Y2','Z2','Weight']
types=['r','r','r','r','r','r','r']
data=[x1,y1,z1,x2,y2,z2,weight]

IAPUC.Generate_IAPUC('edges.txt',data,labels,types)

#write for partiview.

levels=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
color_vals=[2,5,8,11,13,16,18,21,23,25]
for i in range(len(levels)-1):
	weight=weight.astype(float)
	cut_lower=np.where(weight>levels[i])
	cut_upper=np.where(weight<=levels[i+1])
	bins=np.intersect1d(cut_lower,cut_upper)
	f=open('Edges_'+str(levels[i+1])+'.speck','w')
	f.write('#Edges from ('+str(levels[i])+','+str(levels[i+1])+'] \n')
	f.write(' \n ')
	print 'Writing Partiview File: '+str(i+1)+'/10'
	for k in tqdm(range(len(x1[bins]))):
		f.write('mesh -c '+ str(color_vals[i])+' -s wire { \n')
		f.write('1 2 \n')
		f.write(str(x1[bins][k])+' '+str(y1[bins][k])+' '+str(z1[bins][k])+' \n')
		f.write(str(x2[bins][k])+' '+str(y2[bins][k])+' '+str(z2[bins][k])+' \n')
		f.write('} \n')
		f.write(' \n')
	f.close()

#######################################################
# writing the not in groups to file since easiest.
#######################################################

ng_vcmb,ng_zcmb,ng_dist,nx,ny,nz = Funcs.Convert_to_XYZ(l[notgroups],b[notgroups],v[notgroups])

n_K=K[notgroups]-5.*np.log10(ng_dist*1e6)+5   #working out the absolute magnitudes and then calculating a luminoisty scale for partiview files
ng_lum_scale=-1.*(K[notgroups]-5.*np.log10(ng_dist*1e6)+5+20.) #Adding 20 to the current magnitudes and then inverting them to make (most of them) positive
ng_ls_floor=-3. #setting a floor level for the lum_scale (important for partiview to keep some dynamic range)
floor=np.where(ng_lum_scale<ng_ls_floor)[0]   #finding where all the lumscale values are less than the floor 
ng_lum_scale[floor]=ng_ls_floor     		  #assigning those values less than the floor value to the floor value

data=[MASXJIDs[notgroups],Ra[notgroups],Dec[notgroups],l[notgroups],b[notgroups],v[notgroups],ng_vcmb,ng_zcmb,K[notgroups],n_K,ng_lum_scale,ng_dist,nx,ny,nz,other_names[notgroups]]
for i in range(len(data)):  #convert all the arrays into string arrays
	data[i]=data[i].astype(str)

labels=['2MASXJID','Ra','Dec','l','b','vh','vcmb','Zcmb','k_app','K_abs','lum_scale','dist','X','Y','Z','other_names']
units=['s','r','r','r','r','r','r','r','r','r','r','r','r','r','r','s']

IAPUC.Generate_IAPUC(prefix_string+'_notGroups.txt',data,labels,units)

####################################################################################
# Writing the file for subgroups. This will only be the ones that have splintered  #
# The whole data set will be needed however, to do the redshift distortion correction #
########################################################################################

parent_labels=[]
sub_groups=[]
normal_groups=[]
for i in range(len(sub_groupings)):
	if len(sub_groupings[i])>1:
		for k in range(len(sub_groupings[i])):
			sub_groups.append(sub_groupings[i][k])
			parent_labels.append(i+1)
	else:
		normal_groups.append(sub_groupings[i])

subs_l,subs_b,subs_vh=[],[],[]
for i in range(len(sub_groups)):
	subs_l.append(Funcs.WrapMean(l[sub_groups[i]]))
	subs_b.append(np.mean(b[sub_groups[i]]))
	subs_vh.append(np.mean(v[sub_groups[i]]))

subs_l,subs_b,subs_vh=np.array(subs_l),np.array(subs_b),np.array(subs_vh)
subs_vcmb,subs_zcmb,subs_d,subs_x,subs_y,subs_z=Funcs.Convert_to_XYZ(subs_l,subs_b,subs_vh)

##################################################################
# Will do the correction right away, need it for the sub_raddii
##################################################################

subs_radii=[]
subs_corrected_d=[]
subs_galaxies=[]
for i in range(len(sub_groups)):
	group_ra=Funcs.WrapMean(Ra[sub_groups[i]])
	group_dec=np.mean(Dec[sub_groups[i]])
	group_dist=subs_d[i]

	group_members_ra=Ra[sub_groups[i]]
	group_members_dec=Dec[sub_groups[i]]


	seps=Funcs.angsep(group_members_ra,group_ra,group_members_dec,group_dec)		
	Theta=(np.pi/180)*(seps)												
	projected_seps=np.sin(Theta)*(group_dist)							   

	disp=np.std(projected_seps)								

	mu,sigma = 0.,disp 										
	corrected_comoving=np.random.normal(mu,sigma,len(seps))			
	corrected_comoving=group_dist+corrected_comoving

	subs_corrected_d+=list(corrected_comoving)
	subs_radii.append(np.median(projected_seps))   ################ Change over here ####################
	subs_galaxies+=list(sub_groups[i])
subs_galaxies=np.array(subs_galaxies)
subs_galaxies_labels=MASXJIDs[subs_galaxies]


sub_labels=[]
counter=0
parent_labels=np.array(parent_labels)
while counter<len(parent_labels):
	val=np.where(parent_labels==parent_labels[counter])[0]
	subval=np.arange(len(val))+1
	for i in range(len(val)):
		for k in range(len(sub_groups[val[i]])):
			sub_labels.append(str(parent_labels[counter])+'-'+str(subval[i])) 
	counter+=len(val)  


f=open('helloworld.txt','w')
f.write('# 2MRS_label   SubGroupID  \n')
for i in range(len(subs_galaxies_labels)):
	f.write(subs_galaxies_labels[i]+' '+sub_labels[i]+' \n')
f.close()




#################################################################
# Calculating required fields for uncorrected galaxies in groups#
#################################################################

#Calculating the v_cmb
gal_groups=np.concatenate(groups) #gives a list (in order) of all the galaxies which are in groups

gal_groups_vcmb,gal_groups_Zcmb,gal_groups_dist,gal_groups_X,gal_groups_Y,gal_groups_Z = Funcs.Convert_to_XYZ(l[gal_groups],b[gal_groups],v[gal_groups])

#Calculate the Absolite magnitude and the Luminosity scale. 
#since these are extinction corrected magnitudes we don't have an extinction correction
gal_groups_K=K[gal_groups]-5.*np.log10(gal_groups_dist*1e6)+5
gal_groups_lum_scale=-1.*(K[gal_groups]-5.*np.log10(gal_groups_dist*1e6)+5+20.) #Adding 20 to the current magnitudes and then inverting them to make (most of them) positive
gal_groups_ls_floor=-3. #setting a floor level for the lum_scale (important for partiview to keep some dynamic range)
floor=np.where(gal_groups_lum_scale<gal_groups_ls_floor)[0]   #finding where all the lumscale values are less than the floor 
gal_groups_lum_scale[floor]=gal_groups_ls_floor     		  #assigning those values less than the floor value to the floor value



################################################################################################################################################################################
#Generating the first instance of the groupsgal catalog. This can then be read back in to get all the arrays we need as strings and to work out the group data on the spot. 
################################################################################################################################################################################
counter=0 #counter to make sure that entries found in one array eg. gal_groups, dont get repeated in the nested for loop. 
file=open(prefix_string+'_GroupGalaxies.txt','w')
#file.write(\n')
#file.write('[string] \t [deg] \t [deg] \t [deg] \t [deg] \t [km/s] \t [km/s] \t [ap_mag] \t [abs_mag] \t [Mpc] \t [Mpc] \t [Mpc] \t [Mpc] \t [string] \t [deg] \t [deg] \t [deg] \t [deg] \t [km/s] \t [number] \t [number] \n')
for i in range(len(groups)):    #run through all of the groups
	for k in range(len(groups[i])): #go through every galaxy in every group
		file_2masxj=str(MASXJIDs[gal_groups[counter]])+' '
		file_ra=str(Ra[groups[i][k]])+' '
		file_dec=str(Dec[groups[i][k]])+' '
		file_l=str(l[groups[i][k]])+' '
		file_b=str(b[groups[i][k]])+' '
		file_v=str(v[groups[i][k]])+' '
		file_vcmb=str(gal_groups_vcmb[counter])+' '
		file_zcmb=str(gal_groups_Zcmb[counter])+' '
		file_mag=str(K[groups[i][k]])+' '
		file_abs=str(gal_groups_K[counter])+' '
		file_dist=str(gal_groups_dist[counter])+' '
		file_X=str(gal_groups_X[counter])+' '
		file_Y=str(gal_groups_Y[counter])+' '
		file_Z=str(gal_groups_Z[counter])+' '
		file_othernames=str(other_names[gal_groups[counter]])+' '
		file_group_ra=str(Funcs.WrapMean(Ra[groups[i]]))+' '
		file_group_dec=str(np.mean(Dec[groups[i]]))+' '
		file_group_l=str(Funcs.WrapMean(l[groups[i]]))+' '
		file_group_b=str(np.mean(b[groups[i]]))+' '
		file_group_v=str(np.mean(v[groups[i]]))+' '
		file_group_no=str(len(groups[i]))+' '
		file_group_ID=str(i+1)+ ' \n'
		file_string=file_2masxj+file_ra+file_dec+file_l+file_b+file_v+file_vcmb+file_zcmb+file_mag+file_abs+file_dist+file_X+file_Y+file_Z+file_othernames+file_group_ra+file_group_dec+file_group_l+file_group_b+file_group_v+file_group_no+file_group_ID
		file.write(file_string)
		counter+=1  #increasing counter so that the next value in the 1d calculated arrays, eg. gal_groups will move on to the correct next incriment
file.close()

###################################
# Dealing with the Gropup deatails
###################################


#read back in the file and get all the arrays as strings (which we can use to format everything in the correct manner with string widths)
infile=prefix_string+'_GroupGalaxies.txt'
g2masid,gra,gdec,gl,gb,gvh,gvcmb,gzcmb,gk,gK,gd,gX,gY,gZ,gother,Gra,Gdec,Gl,Gb,Gvh,Gno,GID=np.loadtxt(infile,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21),unpack=True,dtype=str)

Gvcmb,GZcmb,Gd,Gx,Gy,Gz = Funcs.Convert_to_XYZ(Gl.astype(float),Gb.astype(float),Gvh.astype(float))


Gvcmb=Gvcmb.astype(str)
GZcmb=GZcmb.astype(str)
Gd=Gd.astype(str)
Gx=Gx.astype(str)
Gy=Gy.astype(str)
Gz=Gz.astype(str)

gal_groups_lum_scale=gal_groups_lum_scale.astype(str)  #make a string to be read in

gal_groups_wc=np.concatenate(weighting).astype(str)
gal_groups_wcn=np.concatenate(weighting_normed).astype(str)

data=[g2masid,gra,gdec,gl,gb,gvh,gvcmb,gzcmb,gk,gK,gal_groups_lum_scale,gd,gX,gY,gZ,gal_groups_wc,gal_groups_wcn,gother,Gra,Gdec,Gl,Gb,Gvh,Gvcmb,GZcmb,Gd,Gx,Gy,Gz,Gno,GID] #a list of arrays which will make writing data easier. 
labels=['2MASXJID','Ra','Dec','l','b','vh','vcmb','Zcmb','k_app','K_abs','lum_scale','dist','X','Y','Z','degree','degree_normed','other_names','Group_Ra','Group_Dec','Group_l','Group_b','Group_vh','Group_vcmb','Group_Zcmb','Group_dist','Group_X','Group_Y','Group_Z','no_Group_Members','Group_ID']
units=['s','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','s','r','r','r','r','r','r','r','r','r','r','r','r','r']

widths=np.zeros(len(data))
for i in range(len(data)):
	max_val_data=len(max(data[i],key=len))+1
	max_val_label=len(labels[i])+1
	if max_val_label>=max_val_data:
		widths[i]=max_val_label
	else:
		widths[i]=max_val_data
widths=widths.astype(int)

IAPUC.Generate_IAPUC(prefix_string+'_GroupGalaxies.txt',data,labels,units)

#############################################################################
# Uncorrected complete, will now create the group catalog (just the groups)
#############################################################################

#just need to find the first instance of each new group.
  #making an int so that the ordering can be ignored with string which would go 1, 10, 100, 1000, 11 etc..
idx=np.unique(GID.astype(int),return_index=True)[1]    #Find where all the new integers are in the labels and return their indecies. 
Gra,Gdec,Gl,Gb,Gvh,Gvcmb,GZcmb,Gd,Gx,Gy,Gz,Gno,GID=Gra[idx],Gdec[idx],Gl[idx],Gb[idx],Gvh[idx],Gvcmb[idx],GZcmb[idx],Gd[idx],Gx[idx],Gy[idx],Gz[idx],Gno[idx],GID[idx] #apply to all the fields
Gdata=data[18:]      #Only take the last few fields which belong to the Groups
Glabels=labels[18:]

Gwidths=widths[18:]

Glabels[8]='X'
Glabels[9]='Y'
Glabels[10]='Z'

units=['r','r','r','r','r','r','r','r','r','r','r','r','r']

f=open(prefix_string+'_Groups.txt','w')
file_string='|'.join('%*s' % p for p in zip(Gwidths,Glabels))
f.write('|'+file_string+'| \n')
file_string='|'.join('%*s' % p for p in zip(Gwidths,units))
f.write('|'+file_string+'| \n')


for i in range(len(Gra)):
	headers=[Gra[i],Gdec[i],Gl[i],Gb[i],Gvh[i],Gvcmb[i],GZcmb[i],Gd[i],Gx[i],Gy[i],Gz[i],Gno[i],GID[i]]
	file_string=' '.join('%*s' % p for p in zip(Gwidths,headers))
	f.write(' '+file_string+' \n')
f.close()

##########################################################################################
# Groups and GalaxyGroups done, using both to generate the redshift distortion correction
##########################################################################################

infile=prefix_string+'_GroupGalaxies.txt'  #reading in the group galaxies file
g2masid,gra,gdec,gl,gb,gvh,gvcmb,gzcmb,gk,gK,gls,gd,gX,gY,gZ,gw,gwn,gother,Gra,Gdec,Gl,Gb,Gvh,Gvcmb,GZcmb,Gd,Gx,Gy,Gz,Gno,GID=np.loadtxt(infile,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30),unpack=True,dtype=str,skiprows=2)

corrected_d=[]
radii=[]
gX=[]
gY=[]
gZ=[]
for i in range(len(idx)):   #idx still represents the places in the main group galaxies catalog where the new groups are
	group_members=np.where(GID.astype(int)==i+1)[0]   #taking advantage of the labels being i+1. This gives us the places where the current group members are
	group_members_ra=gra[group_members].astype(float)		#get the ra's of all the group members
	group_members_dec=gdec[group_members].astype(float)		#get the dec's of all the group members 

	group_ra=float(Gra[idx[i]])									#get the systematic ra of the group
	group_dec=float(Gdec[idx[i]])									#get the systematic dec of the group
	group_dist=float(Gd[idx[i]]) 								    #get the systematic !CMB! velocity of the group

	seps=Funcs.angsep(group_members_ra,group_ra,group_members_dec,group_dec)		#calculate the angular separations (on sky) of all the members from the systematic
	Theta=(np.pi/180)*(seps)												#convert angular separations into radians (to use in trig function) and divide by 2 (instead of dealing with it later)
	projected_seps=np.sin(Theta)*(group_dist)							    #prjected separations at the systemic comoving distance

	disp=np.std(projected_seps)								#get the projected dispersion (assuming a gaussian profile) (using this standard deviation as the characteristic when we draw random samples from a gaussian PDF)

	mu,sigma = 0.,disp 										#set up parameters of the PDF
	corrected_comoving=np.random.normal(mu,sigma,len(group_members))			#generate random comoving
	corrected_comoving=group_dist+corrected_comoving

	corrected_d.append(corrected_comoving)
	radii.append(np.median(projected_seps)) ################ Change over here ######################
	gd[group_members]=corrected_comoving

'''for i in range(len(subs_galaxies_labels)):
	idx=np.where(g2masid==str(subs_galaxies_labels[i]))   ############################## This will convert the distances to the sub groups I THINK ######################################s
	gd[idx]=str(subs_corrected_d[i])'''

radii=np.array(radii).astype(str)
c=SkyCoord(l=gl.astype(float)*u.degree,b=gb.astype(float)*u.degree,distance=gd.astype(float)*u.Mpc,frame='galactic')
gX=c.cartesian.x.value
gY=c.cartesian.y.value
gZ=c.cartesian.z.value


gd=gd.astype(str)  #convert arrays back to strings
gX=gX.astype(str)
gY=gY.astype(str)
gZ=gZ.astype(str)

data=[g2masid,gra,gdec,gl,gb,gvh,gvcmb,gzcmb,gk,gK,gls,gd,gX,gY,gZ,gw,gwn,gother,Gra,Gdec,Gl,Gb,Gvh,Gvcmb,GZcmb,Gd,Gx,Gy,Gz,Gno,GID] #a list of arrays which will make writing data easier. 
labels=['2MASXJID','Ra','Dec','l','b','vh','vcmb','Zcmb','k_app','K_abs','lum_scale','dist','X','Y','Z','degree','degree_normed','other_names','Group_Ra','Group_Dec','Group_l','Group_b','Group_vh','Group_vcmb','Group_Zcmb','Group_dist','Group_X','Group_Y','Group_Z','no_Group_Members','Group_ID']
units=['s','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','s','r','r','r','r','r','r','r','r','r','r','r','r','r']


IAPUC.Generate_IAPUC(prefix_string+'_GroupGalaxies_Corrected.txt',data,labels,units)

#####################################################################
#####################################################################
# Writing the Ascii files to different formats
#####################################################################
#####################################################################


################
# Parti View
###############

#Group Galaxies

infile=prefix_string+'_GroupGalaxies.txt'

g2masid,gra,gdec,gl,gb,gvh,gvcmb,gzcmb,gk,gK,gls,gd,gX,gY,gZ,gw,gwc,gother,Gra,Gdec,Gl,Gb,Gvh,Gvcmb,GZcmb,Gd,Gx,Gy,Gz,Gno,GID=np.loadtxt(infile,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30),unpack=True,dtype=str,skiprows=2)
labels=['Ra','Dec','l','b','vh','vcmb','Zcmb','k_app','K_abs','lum_scale','dist','degree','degree_normed','Group_ID']
data = [gra,gdec,gl,gb,gvh,gvcmb,gzcmb,gk,gK,gls,gd,gw,gwc,GID]

IAPUC.Generate_partiview(prefix_string+'_GroupGalaxies.speck',gX,gY,gZ,data,labels)

##################################################
#############################
#Corrected group galaxies
##########################

infile=prefix_string+'_GroupGalaxies_Corrected.txt'

g2masid,gra,gdec,gl,gb,gvh,gvcmb,gzcmb,gk,gK,gls,gd,gX,gY,gZ,gw,gwn,gother,Gra,Gdec,Gl,Gb,Gvh,Gvcmb,GZcmb,Gd,Gx,Gy,Gz,Gno,GID=np.loadtxt(infile,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30),unpack=True,dtype=str,skiprows=2)
labels=['Ra','Dec','l','b','vh','vcmb','Zcmb','k_app','K_abs','lum_scale','dist','degree','degree_normed','Group_ID']
data = [gra,gdec,gl,gb,gvh,gvcmb,gzcmb,gk,gK,gls,gd,gw,gwn,GID]


IAPUC.Generate_partiview(prefix_string+'_GroupGalaxies_Corrected.speck',gX,gY,gZ,data,labels)
#####################################################
######################
# Not Groups
######################
#### Note that if there is a single entry in not groups.txt, we will have a problem reading it in. 
infile=prefix_string+'_notGroups.txt'
g2masid,gra,gdec,gl,gb,gvh,gvcmb,gzcmb,gk,gK,gls,gd,gX,gY,gZ,gother=np.loadtxt(infile,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),unpack=True,dtype=str,skiprows=2)
labels=['Ra','Dec','l','b','vh','vcmb','Zcmb','k_app','K_abs','lum_scale','dist']
data = [gra,gdec,gl,gb,gvh,gvcmb,gzcmb,gk,gK,gls,gd]

IAPUC.Generate_partiview(prefix_string+'_notGroups.txt',gX,gY,gZ,data,labels)
##############################################################
#################
# Groups
#################

infile=prefix_string+'_Groups.txt'
Gra,Gdec,Gl,Gb,Gvh,Gvcmb,GZcmb,Gd,Gx,Gy,Gz,Gno,GID=np.loadtxt(infile,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12),unpack=True,dtype=str,skiprows=2)

#have to make the fancy mesh pv file. 
tom_radii=np.sqrt(0.96+(Gno.astype(float)/50.))  #scale radius to some value set by Tom
tom_radii=tom_radii.astype(str)

f=open(prefix_string+'_Groups.speck','w')
for i in range(len(Gx)):
	file_string=Gx[i]+' '+Gy[i]+' '+Gz[i]+' ellipsoid -r '+radii[i]+' -c 10 -s wire -n 24 \n' 
	f.write(file_string)
f.close()

f=open(prefix_string+'_Groups_labels.speck','w')
for i in range(len(Gx)):
	file_string=' '+Gx[i]+' '+Gy[i]+' '+Gz[i]+' text '+GID[i]+' \n'
	f.write(file_string)
f.close()
#############################################################

#################
# Sub Groups
#################
#have to make the fancy mesh pv file. 
tom_radii=np.sqrt(0.96+(Gno.astype(float)/50.))  #scale radius to some value set by Tom
tom_radii=tom_radii.astype(str)

f=open(prefix_string+'_SubGroups.speck','w')
for i in range(len(subs_x)):
	file_string=str(subs_x[i])+' '+str(subs_y[i])+' '+str(subs_z[i])+' ellipsoid -r '+str(subs_radii[i])+' -c 20 -s wire -n 24 \n' 
	f.write(file_string)
f.close()


#############################################################


data=[np.array(subs_x).astype(str),np.array(subs_y).astype(str),np.array(subs_z).astype(str),np.array(subs_radii).astype(str)] #a list of arrays which will make writing data easier. 
labels=['X','Y','Z','Radius']
units=['r','r','r','r']

widths=np.zeros(len(data))
for i in range(len(data)):
	max_val_data=len(max(data[i],key=len))+1
	max_val_label=len(labels[i])+1
	if max_val_label>=max_val_data:
		widths[i]=max_val_label
	else:
		widths[i]=max_val_data
widths=widths.astype(int)

f=open(prefix_string+'_SubGroups.txt','w')
file_string='|'.join('%*s' % p for p in zip(widths,labels))
f.write('|'+file_string+'| \n')
file_string='|'.join('%*s' % p for p in zip(widths,units))
f.write('|'+file_string+'| \n')

for i in range(len(subs_x)):
	headers=[subs_x[i],subs_y[i],subs_z[i],subs_radii[i]]
	file_string=' '.join('%*s' % p for p in zip(widths,headers))
	f.write(' '+file_string+' \n')
f.close()
##########################################################################################################
