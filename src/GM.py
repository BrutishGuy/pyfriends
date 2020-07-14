############################################################################
#
# Group Refinement using graph theory
#
###########################################################################


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from astropy import units 
from astropy.coordinates import SkyCoord
import datetime
from tqdm import tqdm


#This module needs to take in a LIST OF ARRAYS. (list of all the various group combinations over several runs)
# when running the algorithm, all results must just be added to a single list. This is the most efficient way to do it. 


def Get_Tupples(array_group):
	group_of_interest=np.sort(array_group) #ordering is very important for this method
	val_x,val_y=[],[]    #populate this empty lists with all possible connections within the local group we are checking. Documenting the galaxies involvement with one another
	for I in range(len(group_of_interest)-1):  #run through the entire group
		for II in range(len(group_of_interest)-1-I): #run through whats left of the array as we carry on checking
			val_x.append(group_of_interest[I])
			val_y.append(group_of_interest[I+II+1])
	return val_x,val_y

def Get_Edges(results_list,cutoff,no_runs):
	print 
	print 'Generating Edges:'
	print '\t 1 of 2: Calculating Pairs'
	edges_x,edges_y=[],[]
	for i in range(len(results_list)):   #start a loading bar for calculating all the different pairings
		tupples=Get_Tupples(results_list[i])   #search through every group that was found in the results and construct the pairings for that group
		edges_x+=tupples[0] 					#add the pairings into 2 lists (x,y) tupple list
		edges_y+=tupples[1]
	edges_x,edges_y=np.array(edges_x),np.array(edges_y)     #convert into arrays (so we can make use of the np.where function)
	all_edges = zip(edges_x,edges_y)

	print '\t 2 of 2: Calculating Weights'
	
	string_tupples = np.array([str(edges_x[i])+' '+str(edges_y[i]) for i in range(len(edges_x))])    #combine the tupples into a single long string array which can be quickly sorted
	tupples, counts = np.unique(string_tupples,return_counts = True)
	Weights = counts.astype(float)/no_runs
	tupple_arrays = np.array([np.array([int(tupples[i].split(' ')[0]),int(tupples[i].split(' ')[1])]) for i in range(len(tupples))]) #convert into 2d array easily split
	Edges_x, Edges_y = tupple_arrays[:,0], tupple_arrays[:,1]
	return Edges_x,Edges_y,Weights

# One line solution to get all the nodes which have shown up. This represents all the galaxies which have been found in a group ever. 
def Get_Nodes(results_list): 
	print 
	print 'Generating Nodes:'
	nodes=np.unique(np.concatenate(results_list))       #first collapse the results so we have one massive array of every result in no order then find the unique numbers. 
	return nodes 										#will be returned as an array 


#function to generate the main graph, applying our cutoff to remove randoms. This graph will then be cut into pieces which will give us our groups
def Generate_Main_Graph(results_list,cutoff,no_runs):
	nodes=Get_Nodes(results_list)								#get the nodes of the graph
	ex,ey,ew=Get_Edges(results_list,cutoff,no_runs)				#get the edges of the graph
	print
	print 'Generating Main Graph:'
	G=nx.Graph()										#generate empty nx graph object
	print '\t 1 of 2: Implementing nodes'
	G.add_nodes_from(nodes)
	print '\t 2 of 2: Implementing edges'
	for i in tqdm(range(len(ex))):
		G.add_edge(ex[i],ey[i],weight=ew[i])
	return G

#very important function which will take in a graph object and return a list of all the subgraphs. 
def Get_Subgraphs(Graph):   #take in a populated nx Graph object  
	sub_graphs=list(Graph.subgraph(c).copy() for c in nx.connected_components(Graph))  #find the connected subgraphs of the inputted graphs and make it a list. 
	return sub_graphs

#function to remove any pairings which are not a group after going through the stability process
def Find_Stable_Groups(sub_graph_list):
	stable_groups=[] 						#list which will be populated with the stable groups (those with 3+ members after all the testing)
	for i in range(len(sub_graph_list)):
		if len(sub_graph_list[i].nodes)>2:
			stable_groups.append(sub_graph_list[i])
	return stable_groups

#function to convert the graph objects into arrays of indicies, as in the exact same input that came in. 
def Get_Node_Arrays(stable_list):   #take in the stable graphs
	Stable_group_galaxies=[]        #make a new list which will be our list of arrays
	for i in range(len(stable_list)):
		Stable_group_galaxies.append(list(stable_list[i].nodes))    #For each graph convert the nodes into a numpy array
	return Stable_group_galaxies

def Get_Edges_Arrays(stable_list):
	edges_array=[]						#start new array which will be populated with our triplates
	for i in range(len(stable_list)):  	 #go through the full list and calculate for each subgraph
		local_edges_full=nx.get_edge_attributes(stable_list[i],'weight')   #generate dictionary of properties of weights
		local_edges_only=list(local_edges_full)								#make a list of the tupples representing the edges
		for edge in range(len(local_edges_only)):
			local_edge_array=[int(local_edges_only[edge][0]),int(local_edges_only[edge][1]),float(local_edges_full[local_edges_only[edge]])] #generate the list with X,Y,Weight(X,Y)
			edges_array.append(local_edge_array)  							#add to master list							
	return np.array(edges_array) 												#return our array

#Function which will take a graph and return the graph with only the edges greater or equal to the inputed threshold value still available. 
def cut_edges(graph,threshold):
	list_graph=[graph] 							#make the graph as a list so that the function doesn't complain
	edge_array=Get_Edges_Arrays(list_graph)     #get the edge array of the graph, i.e. x,y,weight 
	local_nodes=list(graph.nodes())  			#get the nodes of the already existing graph
	new_graph=nx.Graph()    					#make a new graph object
	new_graph.add_nodes_from(local_nodes) 		#Add the nodes from the old graph to the new graph
	for k in range(len(edge_array)): 			#add all the edges. 
		if edge_array[k][-1]>=threshold:    
			new_graph.add_edge(int(edge_array[k][0]),int(edge_array[k][1]),weight=edge_array[k][2])
	return new_graph


#Function to work out both the weighted centrality and the normalized weighted centrality
def Weighted_Centrality(Graph):
	graph_nodes=list(Graph.nodes()) 							#get the nodes of the graph
	sum_weight=np.array(Graph.degree(graph_nodes,'weight'))		#calculate the weighted sum of the edges for each of the nodes
	sum_norm=np.array(Graph.degree(graph_nodes))				#calculate the number of edges to each nodes
	sum_weight=sum_weight[:,1]									#strip off the nodes in the dictionary and retain only the weightings 
	sum_weight_norm=sum_weight.astype(float)/(len(graph_nodes)-1)#sum_norm[:,1] #normalize
	return sum_weight,sum_weight_norm

def WC_list(Graph_list):
	centralities,normed_centralities=[],[]
	for graph in range(len(Graph_list)):
		val=Weighted_Centrality(Graph_list[graph])
		centralities.append(val[0])
		normed_centralities.append(val[1])
	return centralities,normed_centralities


#function to find the rankings of the edges. 
def ranking_Edges(Graph):
	graphs_edges=Get_Edges_Arrays([Graph]) 				#get the edge data as an array of array with u,v,weight
	graphs_edges=np.unique(np.sort(graphs_edges[:,2]))						#get only the weights 
	return graphs_edges 

def SubGroups(Graph):
	edges_rank=ranking_Edges(Graph)    #getting a sorted array of edges which we will cut successively
	number_subgraphs=[]
	subies=[]
	for i in range(len(edges_rank)):
		local_graph=cut_edges(Graph,edges_rank[i])
		local_graph=Get_Subgraphs(local_graph)
		local_graph=Find_Stable_Groups(local_graph)
		number_subgraphs.append(len(local_graph))
		subies.append(local_graph)
	number_subgraphs=np.array(number_subgraphs)
	val_subs=np.where(number_subgraphs==np.max(number_subgraphs))[0][0]
	subies=subies[val_subs]
	subies=Get_Node_Arrays(list(subies))
	return subies


def SubGroups_List(Graph_list):
	sub_groups=[]
	for graph in range(len(Graph_list)):
		sub_groups.append(SubGroups(Graph_list[graph]))
	return sub_groups


#main function which will take in a large list of results after multiple runs and average over everything (using graph theory) then return the specified stable groups.
def Stabalize(results_list,cutoff,no_runs):
	G=Generate_Main_Graph(results_list,cutoff,no_runs)			#first generate the main Graph  (This will remove any weak pairs)
	sub_graphs=Get_Subgraphs(G)									#will allow us to see if any groups fall away
	edge_data=Get_Edges_Arrays(sub_graphs)  					#No cut off has been done just yet. So when we get the edge data this should give us all the edges. 

	#making a cut based on the cut off. 
	G=cut_edges(G,cutoff)
	sub_graphs=Get_Subgraphs(G)	
	stables=Find_Stable_Groups(sub_graphs) 						#cutting out all the unstable groups
	
	
	stable_arrays=Get_Node_Arrays(stables) 						#convert the graphs into arrays		

	#Calculate Graph propeteries (Weightings, Find Subgroups)
	weights,weights_normed=WC_list(stables) 				#getting the weights and the normed weights
	sub_groupings=SubGroups_List(stables) 				#getting the sub groups for each


	return stable_arrays,edge_data,weights,weights_normed,sub_groupings
