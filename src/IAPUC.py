############################################
#
# Script to take arrays and write them as 
# IAPUC Tables 
#
# Trystan Lambert 
# 10/07/2019
#
###########################################

import numpy as np
from astropy.io.votable.tree import VOTableFile,Resource,Table,Field
from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
import os 

def Generate_IAPUC(name,data1,labels,units):
	# calculate the maximum width which will be needed
	data=[]
	for i in range(len(data1)):
		data.append(data1[i].astype(str))

	widths=np.zeros(len(data))
	for i in range(len(data)):
		max_val_data=len(max(data[i],key=len))+1
		max_val_label=len(labels[i])+1  			#takes into account labels too
		max_val_units=len(units[i])+1  				#and units
		if max_val_label>=max_val_data and max_val_label>=max_val_units:
			widths[i]=max_val_label
		elif max_val_data>=max_val_label and max_val_data >= max_val_units:
			widths[i]=max_val_data
		else:
			widths[i]=max_val_units
	widths=widths.astype(int)

	#write the headers of the table
	f=open(name,'w')
	file_string='|'.join('%*s' % p for p in zip(widths,labels))
	f.write('|'+file_string+'| \n')
	file_string='|'.join('%*s' % p for p in zip(widths,units))
	f.write('|'+file_string+'| \n')

	#write the body of the table
	for i in range(len(data[0])):
		headers=[]
		for k in range(len(data)):
			headers.append(data[k][i])
		file_string=' '.join('%*s' % p for p in zip(widths,headers))
		f.write(' '+file_string+' \n')
	f.close()


def Generate_partiview(name,x_array1,y_array1,z_array1,data1,labels):
	data=[]
	for i in range(len(data1)):
		data.append(data1[i].astype(str))
	x_array=x_array1.astype(str)
	y_array=y_array1.astype(str)
	z_array=z_array1.astype(str)

	f=open(name,'w')
	f.write('datavar 0 texnum \n')
	for i in range(len(labels)):
		f.write('datavar '+str(i+1)+' '+labels[i]+' \n')
	f.write('texturevar texnum \n')
	f.write('texture -M 1 halo.rgb \n')

	for k in range(len(data[0])):
		f.write(x_array[k]+' '+y_array[k]+' '+z_array[k]+' 1 ')
		for i in range(len(data)):
			f.write(data[i][k]+' ')
		f.write(' \n')
	f.close()

def Generate_VO(name,data1,labels):
	data=Table(rows=data1,names=labels)
	ascii.write(data,'values.tbl',format='ipac',overwrite=True)
	t=Table.read('values.tbl',format='ipac')
	t.write(name+'.xml',format='votable',overwrite=True)
	os.remove('values.tbl')

#must be in the correct format for this to work properly #two rows with the first one being the headers 
def Convert_IAPUC_to_VO(name):
	f=open(name)
	lines=[]
	for line in f:
		lines.append(line)
	labels=lines[0].split('|')
	f.close()
	
	header=[]
	for i in range(len(labels)):
		if len(labels[i].strip())>0:
			header.append(labels[i].strip())

	Data=np.loadtxt(name,dtype=str,skiprows=2)

	new_name=name.split('.')  ##this will cause errors if file name has more than one dot !!
	if len(new_name)>2:
		print 'In Future Please reserve . for file extenstions IDIOT!'
		new_new_name=[]
		for i in range(len(new_name)-1):
			new_new_name+=new_name[i]
		new_name=new_new_name
	else:
		new_name=new_name[0]

	Generate_VO(new_name,Data,header)










