import numpy as np
import gmsh
import ufl
import dolfinx
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI #import parallel communicator

from eitx.utils import theta

class MeshClass:
	def __init__(self,electrodes, mesh_refining=1,bdr_refining=1):
		self.electrodes = electrodes
		self.mesh = self.setup_mesh(mesh_refining,bdr_refining).mesh    
		self.ds = self.setup_integration_domain()
	
	def setup_mesh(self,mesh_refining=1,bdr_refining=1):

		electrodes = self.electrodes
		# if running again, you must remove the comment in the following
		gmsh.finalize()
		gmsh.initialize()
		disk = gmsh.model.occ.addDisk(0, 0, 0, 1, 1) #creates disk centered in (0,0,0) with axis (1,1)

		mesh_comm = MPI.COMM_WORLD
		gmsh_model_rank = 0

		#create point list for electrodes
		electrodes_points = []
		for electrode in electrodes.position:
			theta_array = np.linspace(electrode[0],electrode[1],)
			electrodes_points.extend([gmsh.model.occ.addPoint(np.cos(theta),np.sin(theta),0) for theta in theta_array])

		gmsh.model.occ.synchronize()
		gdim = 2 #variable to control disk dimension, where 2 stands for surface
		gmsh.model.addPhysicalGroup(gdim, [disk], 1) # starts mesh object
		# gmsh.model.addPhysicalGroup(0, electrodes_points, 2) #electrodes

		gmsh.option.setNumber("Mesh.CharacteristicLengthMax",0.2 * mesh_refining) # control max length of cells
		gmsh.model.mesh.field.add("Distance", 1)
		gmsh.model.mesh.field.setNumbers(1, "PointsList", electrodes_points)
		gmsh.model.mesh.field.add("Threshold", 2)
		gmsh.model.mesh.field.setNumber(2, "InField", 1)
		gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.03 * bdr_refining)
		gmsh.model.mesh.field.setNumber(2, "SizeMax", 0.25 * bdr_refining )
		gmsh.model.mesh.field.setNumber(2, "DistMin", 0.075 * bdr_refining )
		gmsh.model.mesh.field.setNumber(2, "DistMax", 0.1 * bdr_refining)
		gmsh.model.mesh.field.add("Min", 3)
		gmsh.model.mesh.field.setNumbers(3, "FieldsList", [2])
		gmsh.model.mesh.field.setAsBackgroundMesh(3)
		gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
		gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
		gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
		gmsh.option.setNumber("Mesh.Algorithm", 5)
		gmsh.model.mesh.generate(gdim)

		return gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)

	def setup_integration_domain(self):
		electrodes = self.electrodes
		tol = 0.01 # tolerance for checking if in electordes
		L = electrodes.L

		#setting boundaries markers and indicator functions
		boundaries = [
				(i, lambda x,i=i: np.where(np.logical_and(theta(x)>=electrodes.position[i][0]-tol,theta(x)<=electrodes.position[i][1]+tol),1,0))
				for i in range(L)
		]
		#creating facet tags
		bdr_facet_indices, bdr_facet_markers = [], []
		for (marker, locator) in boundaries:
				facets = dolfinx.mesh.locate_entities_boundary(self.mesh, self.mesh.topology.dim - 1, locator)
				bdr_facet_indices.append(facets)
				bdr_facet_markers.append(np.full_like(facets, marker))
		bdr_facet_indices = np.hstack(bdr_facet_indices).astype(np.int32)
		bdr_facet_markers = np.hstack(bdr_facet_markers).astype(np.int32)
		sorted_facets = np.argsort(bdr_facet_indices)
		facet_tag = dolfinx.mesh.meshtags(self.mesh, self.mesh.topology.dim - 1, bdr_facet_indices[sorted_facets], bdr_facet_markers[sorted_facets])

		return ufl.Measure("ds", domain=self.mesh, subdomain_data=facet_tag)