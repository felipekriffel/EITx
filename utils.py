import numpy as np
import dolfinx
import ufl

def current_method(L,l, method=1, value=1):
	"""
	Create a numpy array (or a list of arrays) that represents the current pattern in the electrodes.

	:param L: Number of electrodes.
	:type L: int
	:param l: Number of measurements.
	:type l: int
	:param method: Current pattern. Possible values are 1, 2, 3, or 4 (default=1).
	:type method: int
	:param value: Current density value (default=1).
	:type value: int or float

	:returns: list of arrays or numpy array -- Return list with current density in each electrode for each measurement.

	:Method Values:
			1. 1 and -1 in opposite electrodes.
			2. 1 and -1 in adjacent electrodes.
			3. 1 in one electrode and -1/(L-1) for the rest.
			4. For measurement k, we have: (sin(k*2*pi/16) sin(2*k*2*pi/16) ... sin(16*k*2*pi/16)).

	:Example:

	Create current pattern 1 with 4 measurements and 4 electrodes:

	>>> I_all = current_method(L=4, l=4, method=1)
	>>> print(I_all)
			[array([ 1.,  0., -1.,  0.]),
			array([ 0.,  1.,  0., -1.]),
			array([-1.,  0.,  1.,  0.]),
			array([ 0., -1.,  0.,  1.])]

	Create current pattern 2 with 4 measurements and 4 electrodes:

	>>> I_all = current_method(L=4, l=4, method=2)
	>>> print(I_all)
			[array([ 1., -1.,  0.,  0.]),
			array([ 0.,  1., -1.,  0.]),
			array([0.,  0.,  1., -1.]),
			array([ 1.,  0.,  0., -1.])]

	"""
	I_all=[]
	#Type "(1,0,0,0,-1,0,0,0)"
	if method==1:
			if L%2!=0: raise Exception("L must be odd.")

			for i in range(l):
					if i<=L/2-1:
							I=np.zeros(L)
							I[i], I[i+int(L/2)]=value, -value
							I_all.append(I)
					elif i==L/2:
							print("This method only accept until L/2 currents, returning L/2 currents.")
	#Type "(1,-1,0,0...)"
	if method==2:
			for i in range(l):
					if i!=L-1:
							I=np.zeros(L)
							I[i], I[i+1]=value, -value
							I_all.append(I)
					else:
							I=np.zeros(L)
							I[0], I[i]=-value, value
							I_all.append(I)
	#Type "(1,-1/15, -1/15, ....)"
	if method==3:
			for i in range(l):
					I=np.ones(L)*-value/(L-1)
					I[i]=value
					I_all.append(I)
	#Type "(sin(k*2*pi/16) sin(2*k*2*pi/16) ... sin(16*k*2*pi/16))"
	# if method==4:
	#     for i in range(l):
	#         I=np.ones(L)
	#         for k in range(L): I[k]=I[k]*sin((i+1)*(k+1)*2*pi/L)
	#         I_all.append(I)

	return I_all

def get_boundary_data(u):
	"""
	Returns an array of function values on boundary, ordered by the angle theta
	"""
	V = u.function_space
	domain = V.mesh
	domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
	boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
	boundary_dofs_index_array = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim-1, boundary_facets) #array with the vertices index
	#gets x and y coordinates for the boundary
	dofs_coordinates = V.tabulate_dof_coordinates()
	x_bdr = dofs_coordinates[boundary_dofs_index_array][:,0]
	y_bdr = dofs_coordinates[boundary_dofs_index_array][:,1]

	#gets the t in [0,2pi] from the corresponding (x,y) coordinates
	#next, gets the index of the sorted t array
	theta = np.where(y_bdr>=0,np.arccos(x_bdr),2*np.pi - np.arccos(x_bdr))
	sorted_theta_index = np.argsort(theta)
	return u.x.array[boundary_dofs_index_array][sorted_theta_index]

def compute_gradient(u: dolfinx.fem.Function):
	"""
	Compute gradient of some tent space function
	Returns coordinates of gradient in each triangle, with array order
	beeing the same of the corresponding indicator function space
	"""

	mesh = u.function_space.mesh
	adjacency_list = mesh.topology.connectivity(2,0)
	mesh_vertex_index_list = []
	for i in range(len(adjacency_list)):
		mesh_vertex_index_list.append(adjacency_list.links(i))

	cells_coordinates_list = []
	for cell_vertex_index in mesh_vertex_index_list:
		cells_coordinates_list.append(mesh.geometry.x[cell_vertex_index][:,:2])

	gradient_list = []

	for idx in mesh_vertex_index_list:
		cell_coord = mesh.geometry.x[idx][:,:2]
		system_array = np.concatenate((cell_coord,np.ones(3).reshape(3,1)),axis=1)
		u_t = u.x.array[idx]
		plane_coef = np.linalg.solve(system_array, u_t)
		gradu = plane_coef[:2]
		# gradu_x, gradu_y = plane_coef[0],plane_coef[1]

		gradient_list.append(gradu)

	return np.array(gradient_list)

# gets theta of (x,y) in polar coord. (r,theta), with theta in interval [0,2pi]
def theta(x):
	r = (x[0]**2 + x[1]**2)**(0.5)
	inv_r = np.where(np.isclose(r,0),0,1/r)
	return np.where(x[1]>=0,np.arccos(x[0]*inv_r),2*np.pi - np.arccos(x[0]*inv_r))

def funcProduct(u,v):
	product = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(u,v) * ufl.dx))
	return product

def funcSquareNorm(u):
	return dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(u,u) * ufl.dx))

def ConvertingData(U,method):
	"""
	Convert data from different measurement patterns to the ground pattern.

	:param U: The data to be converted.
	:type U: numpy.ndarray
	:param method: The measurement pattern to be converted to. Currently only "KIT4" is supported.
	:type method: str

	:return: The converted data.
	:rtype: numpy.ndarray
	"""
	if method=="KIT4": 
		#See: https://arxiv.org/pdf/1704.01178.pdf
		L=len(U)
		U_til = np.zeros(L)
		for i in range(1,L): U_til[i]=np.sum(U[:i])
		c = np.sum(U_til)
		return c/L-U_til
	
	if method=="adjacent":
		U_til = np.zeros_like(U)
		U_til = U - np.roll(U,-1,axis=1)
	return U_til


def GammaCircle(V0, in_v, out_v, radius,centerx, centery):
	"""
	Function to create a circle in the mesh with specified properties.

	:param V0: FunctionSpace.
	:type mesh: :class:`dolfin.cpp.mesh.Mesh`
	:param in_v: Value inside the circle.
	:type in_v: float
	:param out_v: Value outside the circle.
	:type out_v: float
	:param radius: Circle radius.
	:type radius: float
	:param centerx: Circle center position x.
	:type centerx: float
	:param centery: Circle center position y.
	:type centery: float
	
	:returns:  numpy.array -- Return a vector where each position corresponds to the value of the function in that element.
	
	:Example:

	>>> ValuesCells0 = GammaCircle(mesh=mesh_refined, in_v=3.0, out_v=1.0, radius=0.50, centerx=0.25, centery=0.25)
	>>> print(ValuesCells0)
			array([1., 1., 1., ..., 1., 1., 1.])
	>>> Q = FunctionSpace(mesh, "DG", 0) #Define Function space with basis Descontinuous Galerkin
	>>> gamma = Function(Q)
	>>> gamma.vector()[:]=ValuesCells0
	>>> plot_figure(gamma, name="", map="jet");
	
	.. image:: codes/gamma.png
			:scale: 75 %
	"""
	# for i in range(0, mesh.num_cells()):
	#     cell = Cell(mesh, i) #Select cell with index i in the mesh.
			
	#     vertices=np.array(cell.get_vertex_coordinates()) #Vertex cordinate in the cell.
	#     x=(vertices[0]+vertices[2]+vertices[4])/3           
	#     y=(vertices[1]+vertices[3]+vertices[5])/3
			
	#     #If the baricenter is outside the circle...
	#     if ((x-centerx)**2+(y-centery)**2>=radius**2):
	#         ValuesGamma[i]=out_v
	#     else:
	#         ValuesGamma[i]=in_v
	gamma_locator = getGammaCircleLocator(radius,centerx,centery)
	gamma = dolfinx.fem.Function(V0)
	gamma_cells = gamma_locator(V0.tabulate_dof_coordinates().T).astype(int)
	gamma.x.array[:] = out_v + (in_v - out_v)*gamma_cells
	
	return gamma

def getGammaCircleLocator(radius,centerx, centery):
	gammaLocator = lambda x: (x[0]-centerx)**2 + (x[1]-centery)**2<=radius**2
	return gammaLocator