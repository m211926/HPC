import numpy as np
import h5py

def writeH5(s, r, filename):
  """
  Write the h5 file that will save the information needed in proper structure.
  s = numpy array with state data
  r = numpy array with rank data	
  filename = string with desired filename
  """

  f = h5py.File("./parafiles/" + filename,'w')

  # Store state and rank data into the state and rank group of h5 file
  state_group = f.create_group("state_group")
  state = state_group.create_dataset("state",data=s)

  rank_group = f.create_group("rank_group")
  rank = rank_group.create_dataset("rank",data=r)

  f.close()

def writeXdmf(dims,dx,filename,h5_file):
  """
  Write the xmf file, that describes the hdf5 data, to be read by Paraview.
  filename = string with the desired filename
  dims = 3-tuple with the number of rank in each dimension (z,y,x)
  """

  f = open("./parafiles/" + filename,'w')
  f.write('<?xml version="1.0" ?>\n')
  f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
  f.write('<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.1">\n')
  f.write('<Domain>\n')

  f.write('<Grid Name="my_Grid" GridType="Uniform">\n')
  f.write('<Topology TopologyType="3DCoRectMesh" Dimensions="%i %i %i">\n'%(dims[0],dims[1],dims[2]))
  f.write('</Topology>\n')

  f.write('<Geometry GeometryType="Origin_DxDyDz">\n')
  f.write('<DataItem Dimensions="3" NumberType="Integer" Format="XML">\n')
  f.write('0 0 0\n') 
  f.write('</DataItem>\n')
  f.write('<DataItem Dimensions="3" NumberType="Integer" Format="XML">\n')
  f.write('%g %g %g\n'%(dx,dx,dx))
  f.write('</DataItem>\n')
  f.write('</Geometry>\n')

  f.write('<Attribute Name="state" AttributeType="Scalar" Center="Node">\n')
  f.write('<DataItem Dimensions="%i %i %i" NumberType="Integer" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
  f.write('%s:/state_group/state\n'%h5_file)
  f.write('</DataItem>\n')
  f.write('</Attribute>\n')

  f.write('<Attribute Name="rank" AttributeType="Scalar" Center="Node">\n')
  f.write('<DataItem Dimensions="%i %i %i" NumberType="Integer" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
  f.write('%s:/rank_group/rank\n'%h5_file)
  f.write('</DataItem>\n')
  f.write('</Attribute>\n')

  f.write('</Grid>\n')
  f.write('</Domain>\n')
  f.write('</Xdmf>\n')

  f.close()

