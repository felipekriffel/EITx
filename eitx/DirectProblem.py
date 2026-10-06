import dolfinx
from mpi4py import MPI #import parallel communicator
import numpy as np
import ufl
import dolfinx.fem.petsc
import scipy as sp
import basix
from petsc4py import PETSc #Linear algebra lib
from eitx.utils import compute_gradient
import eitx.MeshClass as MeshClass

class DirectProblem:
	"""
	Class for handling Direct Problem. It provides the setup with mesh, 
  
	"""
	def __init__(self, mesh_object: MeshClass, z_values):
		self.mesh_object = mesh_object
		self.mesh = mesh_object.mesh
		self.ds = mesh_object.ds
		self.L = mesh_object.electrodes.L

		Ve = basix.ufl.element('Lagrange', "triangle", degree=1, shape=())
		V0e = basix.ufl.element('Discontinuous Lagrange', "triangle", degree=0, shape=())

		self.V0 = dolfinx.fem.functionspace(self.mesh, V0e)
		self.V = dolfinx.fem.functionspace(self.mesh, Ve)

		self.N = self.V.dofmap.index_map.size_global #number of vertices
		self.K = self.V0.dofmap.index_map.size_global #number of triangles

		self.assembled = False
		self.solved = False
		self.adjoint_set = False
		self.z_values = z_values
		self.z_petsc_array = [dolfinx.fem.Constant(self.mesh, PETSc.ScalarType(z_value)) for z_value in self.z_values]
		self.rtol = 1e-10

		self.set_submatrices()

	def set_submatrices(self):
		v = ufl.TestFunction(self.V)
		#A2
		A2_form_array = [-ufl.inner((1/self.z_petsc_array[i]), v) * self.ds(i) for i in range(self.L)]
		A2_form_obj = [dolfinx.fem.form(form) for form in A2_form_array]
		A2_vector_array= [dolfinx.fem.petsc.assemble_vector(form) for form in A2_form_obj]
		self.A2_np_matrix = np.stack(
		[A2_vector_array[j].getValues(range(self.N)) for j in range(self.L)],
		axis=1
		)
		
		#Assemble A4
		cell_area_list = []
		for i in range(self.L):
			v = ufl.TrialFunction(self.V0) #trial object
			s = v*self.ds(i) # creates object of the form \int_{e_i} 1*ds
			cell_area_form = dolfinx.fem.form(s)
			cell_area = dolfinx.fem.assemble_scalar(cell_area_form)
			cell_area_list.append(cell_area.real)
   
		cell_area_array = np.array(cell_area_list)
		self.A4_np_matrix = np.diag(cell_area_array*(1/self.z_values))


	def set_problem(self, gamma):
		"""
		Assemble operator matrix based on gamma informed
		Matrix A^H@A is saved in scipy csr sparse type object

		Params:
		:gamma: dolfinx.fem.Function(V0), admitivity function object
		"""
		V0 = self.V0
		self.gamma = gamma

		u = ufl.TrialFunction(self.V) #function u_h
		v = ufl.TestFunction(self.V)
		gradu = ufl.grad(u)
		gradv = ufl.grad(v)
		
		#A1
		A1_sum_array = [
		(1/self.z_petsc_array[i])*ufl.inner(u,v)*self.ds(i) for i in range(self.L)
		]
		A1_form = (self.gamma * ufl.inner(gradu, gradv)) * ufl.dx + np.sum(A1_sum_array)
		A1_form_obj = dolfinx.fem.form(A1_form)
		A1_matrix = dolfinx.fem.petsc.assemble_matrix(A1_form_obj)
		A1_matrix.assemble()
		
		A1_np_matrix = A1_matrix.getValues(range(self.N),range(self.N))
		self.A_np = np.block([[A1_np_matrix, self.A2_np_matrix],
						[self.A2_np_matrix.T, self.A4_np_matrix]])

		self.A_op = sp.sparse.csr_matrix(self.A_np.T.conj()@self.A_np)
		self.A_conj_op = self.A_op.conjugate()

		self.assembled = True


	def solve_problem_current(self, I_all,gamma=None,save=False):
		"""
		Solve the normal problem (A^h A) u = (A^h) b, given the I_all current array
		Matrix needs to have been assembled through method `set_problem`

		Params:
		:I_all: list of current patterns
		:save: Boolean, save solution

		Returns:
		:u_list: list of dolfinx.fem.Function objects, for each potentital distribution
		:U_list: list of np.array objects, for each measured potential pattern array
		"""
		if gamma!=None:
			self.set_problem(gamma)

		if not self.assembled:
			raise Exception("The problem operator has not been assembled")

		u_list = []
		u_array_list = []
		U_list = []

		l = len(I_all)

		for k in range(l):
			b = np.block([np.zeros(self.N,dtype=PETSc.ScalarType),I_all[k]])
   
		u_nest = sp.sparse.linalg.spsolve(self.A_op,self.A_np.T.conj()@b) #Solve A^*Au = b
		# u_nest, err_code = sp.sparse.linalg.cg(self.A_op,self.A_np.T.conj()@b,rtol=self.rtol) #Solve A^*Au = b
		
		u_array, U_array = u_nest[:self.N], u_nest[self.N:] #splitting array

		# translating solutions (U = (U1 + S/L,...,UL + S/L)), with S = U1+...+UL
		S = U_array.sum()
		U_array -= S/self.L
		u_array -= S/self.L

		u_array_list.append(u_array)
		U_list.append(U_array)

		for u_array in u_array_list:
			u = dolfinx.fem.Function(self.V)
			u.x.array[:] = u_array
			u_list.append(u)

		if save:
			self.u_list = u_list
			self.U_list = U_list

		return u_list, U_list


	def solve_problem_vector(self,vector_list, save=False):
		"""
		Solve the normal problem (A^h A) u = (A^h) b, given the b vector list
		Useful for solving derivative problem, for example
		Matrix needs to have been assembled through method `set_problem`

		Params:
		:vector_list: list of current patterns

		Returns:
		:u_list: list of dolfinx.fem.Function objects, for each potentital distribution
		:U_list: list of np.array objects, for each measured potential pattern array
		"""
		if not self.assembled:
			raise Exception("The problem operator has not been assembled")

		u_list = []
		u_array_list = []
		U_list = []

		for b in vector_list:
			u_nest = sp.sparse.linalg.spsolve(self.A_op,self.A_np.T.conj()@b) #Solve A^*Au = b
			# u_nest, err_code = sp.sparse.linalg.cg(self.A_op,self.A_np.T.conj()@b,rtol=self.rtol) #Solve A^*Au = b
	  
		u_array, U_array = u_nest[:self.N], u_nest[self.N:] #splitting array

		# translating solutions (U = (U1 + S/L,...,UL + S/L)), with S = U1+...+UL
		S = U_array.sum()
		U_array -= S/self.L
		u_array -= S/self.L

		u_array_list.append(u_array)
		U_list.append(U_array)

		for u_array in u_array_list:
			u = dolfinx.fem.Function(self.V)
			u.x.array[:] = u_array
			u_list.append(u)

		if save:
			self.u_list = u_list
			self.U_list = U_list

		return u_list, U_list

	def directional_derivative(self,eta,u_list=None,use_stored_u=False):
		if use_stored_u:
			if not self.solved:
				raise Exception("There is no stored solution yet")

		u_list = self.u_list

		#Create Trial and test functions
		w = ufl.TrialFunction(self.V) #function u_h
		v = ufl.TestFunction(self.V)

		L_array = []

		#creates object representing the gradient of w, us and v functions
		gradv = ufl.grad(v)
		for us in u_list:
			gradus = ufl.grad(us)

		#creating form
		L_exp = - (eta * ufl.inner(gradus,gradv)) * ufl.dx
		L_form = dolfinx.fem.form(L_exp)

		#getting rhs vector
		L_vector = dolfinx.fem.petsc.assemble_vector(L_form)
		L_nparray = L_vector.getValues(range(L_vector.getSize()))

		L_array.append(np.block([L_nparray,np.zeros(self.L)]))


		ws, Ws = self.solve_problem_vector(L_array)
		return ws,Ws

	def adjoint(self, sigma_list, u_list):
		"""
		Get sigma_list direction, returns F'(gamma)* sigma_list
		"""
		psi_list, Psi_list  = self.solve_adjoint_problem(sigma_list)
		grad_psi_list = [compute_gradient(psi) for psi in psi_list]
		grad_u_list = [compute_gradient(u) for u in u_list]
		adj = dolfinx.fem.Function(self.V0)
		adj_j_array = np.zeros_like(adj.x.array)

		l = len(u_list)
		for j in range(l):
			graduj = grad_u_list[j]
		
		gradpsij = grad_psi_list[j]

		for k in range(adj_j_array.shape[0]):
			adj_j_array[k] = - np.vdot(graduj[k],gradpsij[k])

		adj.x.array[:] += adj_j_array

		return adj

	def solve_adjoint_problem(self, I_all):
		if not self.assembled:
			raise Exception("The problem operator has not been assembled")

		u_list = []
		u_array_list = []
		U_list = []

		l = len(I_all)

		for k in range(l):
			b = np.block([np.zeros(self.N),I_all[k]])
   
		u_nest = sp.sparse.linalg.spsolve(self.A_conj_op,self.A_np.T@b) #Solve A^*Au = b
		u_array, U_array = u_nest[:self.N], u_nest[self.N:] #splitting array

		# translating solutions (U = (U1 + S/L,...,UL + S/L)), with S = U1+...+UL
		S = U_array.sum()
		U_array -= S/self.L
		u_array -= S/self.L

		u = dolfinx.fem.Function(self.V)
		u.x.array[:] = u_array
		u_list.append(u)

		U_list.append(U_array)

		return u_list, U_list
		