import numpy as np
import eitx.DirectProblem as DirectProblem
import dolfinx
from mpi4py import MPI #import parallel communicator
import numpy as np
import ufl
import dolfinx.fem.petsc
from dolfinx.io import gmsh as gmshio
import scipy as sp
import basix
from petsc4py import PETSc #Linear algebra lib

from eitx.utils import compute_gradient

class InverseProblem(DirectProblem):
	def __init__(self,mesh_inverse, z_values,I_all):
		super().__init__(mesh_inverse,z_values)
	
		#"First guess and weight functions"
		self.firstguess_array = np.full(self.K,1+1j)        #First guess for Forwardproblem
		self.weight=np.eye(self.K)             #Initial weight function
		self.I_all = I_all
		
		#"Solver configurations"
		self.verbose=False
		self.weight_value=False    #Are you going to use the weight function in the Jacobian matrix?
		self.weight_type = "Area" #"Area" for only area inverses, "Jacobian" for area and Jacobian column norm weights
		self.step_limit=30        #Step limit while solve
		self.innerstep_limit=1000 #Inner step limit while solve
		self.min_v=1E-3           #Minimal value in element for gamma_k
		
		#"Noise Configuration"
		self.noise_level=0      #Noise_level from data (%) Ex: 0.01 = 1%
		self.tau=1.01           #Tau for disprance principle, tau>1
		
		#"Newton parameters"
		self.mu_i=0.9       #Mu initial (0,1]
		self.mumax=0.999    #Mu max
		self.nu=0.99        #Decrease last mu_n
		self.R=0.98         #Maximal decrease (%) for mu_n
		
		#"Inner parameters"
		self.inner_method='Landweber'  # Default inner method for solve Newton
		
		#Other Default parameters
		self.land_a=1    #Step-size Landweber
		self.ME_reg=5E-4 #Regularization Minimal Error
		self.Tik_c0=1    #Regularization parameter Iterative Tikhonov
		self.Tik_q=0.95  #Regularization parameter Iterative Tikhonov
		self.LM_c0=1     #Regularization parameter Levenberg-Marquadt
		self.LM_q=0.95   #Regularization parameter Levenberg-Marquadt
		
		#"A priori information"
		self.gamma_sol = None   #Exact Solution
		self.mesh_sol = None       #Mesh of exact solution
		self.gamma_inv = None   #Interpolated solution

		# Jacobian setup
		v = ufl.TrialFunction(self.V0)
		cell_area_form = dolfinx.fem.form(v*ufl.dx)
		cell_area = dolfinx.fem.assemble_vector(cell_area_form)
		self.cell_area_array = cell_area.array.real
		self.id_matrix = np.eye(self.L) - 1/self.L
		
	def set_answer(self, gamma0):
		"""
		Set the exact solution for gamma.

		This method sets the exact solution (gamma0) and its corresponding mesh (mesh0) to be used for comparison and error calculation. This is useful to determine the best solution reached.

		:param gamma0: Finite Element Function representing the exact solution for gamma.
		:type gamma0: Function

		:Example:
		>>> InverseObject=InverseProblem(mesh_inverse, list_U0_noised, I_all, z)
		>>> InverseObject.set_answer(gamma0)

		"""
		mesh_dir = gamma0.function_space.mesh
		gamma_inv = dolfinx.fem.Function(self.V0)
		gamma_inv.interpolate(gamma0,nmm_interpolation_data=dolfinx.fem.create_nonmatching_meshes_interpolation_data(
			self.mesh,
			self.V0.element,
			mesh_dir)
		)
		self.gamma_inv = gamma_inv
		self.gamma_sol = gamma0

	def set_NewtonParameters(self,  **kwargs):
		"""Set Newton Parameters for the inverse problem.

		Kwargs:
				* **mu_i** (float): Initial value for mu (0, 1].
				* **mumax** (float): Maximum value for mu (0, 1].
				* **nu** (float): Factor to decrease the last mu_n.
				* **R** (float): Minimal value for mu_n.

		Default Parameters:
				>>> self.mu_i = 0.9
				>>> self.mumax = 0.999
				>>> self.nu = 0.99
				>>> self.R = 0.9

		:Example:
		>>> InverseObject = InverseProblem(mesh_inverse, list_U0_noised, I_all, l, z)
		>>> InverseObject.set_NewtonParameters(mu_i=0.90, mumax=0.999, nu=0.985, R=0.90)
		"""
		for arg in kwargs:
				setattr(self, arg, kwargs[arg])
		return

	def set_NoiseParameters(self, tau, noise_level):
		"""
		Set Noise Parameters for stopping with the Discrepancy Principle.

		:param tau: Tau value for the discrepancy principle [0, ∞).
		:type tau: float
		:param noise_level: Noise level (%) in the data [0, 1).
		:type noise_level: float

		Default Parameters:
				>>> self.tau = 0
				>>> self.noise_level = 0

		:Example:
		>>> InverseObject = InverseProblem(mesh_inverse, list_U0_noised, I_all, l, z)
		>>> InverseObject.set_NoiseParameters(tau=5, noise_level=0.01)
		"""

		self.tau=tau
		self.noise_level=noise_level

	def set_solverconfig(self, **kwargs):
		"""Set Solver configuration for the inverse problem.

		Kwargs:
				* **weight_value** (bool): Use a weight function in the Jacobian matrix.
				* **step_limit** (float): Step limit while solving.
				* **min_v** (float): Minimal value in an element for gamma_k.

		Default Parameters:
				>>> self.weight_value = True
				>>> self.step_limit = 5
				>>> self.min_v = 0.05

		:Example:
		>>> InverseObject = InverseProblem(mesh_inverse, list_U0_noised, I_all, l, z)
		>>> InverseObject.set_solverconfig(weight_value=True, step_limit=200, min_v=0.01)
		"""    
		for arg in kwargs:
				setattr(self, arg, kwargs[arg])

	def set_InnerParameters(self, **kwargs):
		"""Set Inner-step Newton Parameters for the inverse problem.

		Kwargs:
				* **inner_method** (str): Method to solve the inner step Newton. Options: 'Landweber', 'CG', 'ME', 'LM', 'Tikhonov'.
				* **land_a** (int): Step-size for Landweber method.
				* **ME_reg** (float): Minimal value in an element for gamma_k.
				* **Tik_c0** (float): Regularization parameter for Iterative Tikhonov.
				* **Tik_q** (float): Regularization parameter for Iterative Tikhonov.
				* **LM_c0** (float): Regularization parameter for Levenberg-Marquardt.
				* **LM_q** (float): Regularization parameter for Levenberg-Marquardt.

		Default Parameters:
				>>> self.inner_method = 'Landweber'
				>>> self.land_a = 1
				>>> self.ME_reg = 5E-4
				>>> self.Tik_c0 = 1
				>>> self.Tik_q = 0.95
				>>> self.LM_c0 = 1
				>>> self.LM_q = 0.95

		:Example:
		>>> InverseObject = InverseProblem(mesh_inverse, list_U0_noised, I_all, l, z)
		>>> InverseObject.set_InnerParameters(inner_method='ME', ME_reg=5E-4)
		"""
		for arg in kwargs:
				setattr(self, arg, kwargs[arg])


	def solve_inverse(self, U_list):
		"""
			Solve inverse problem for the given U_list data

		Returns (tuple):
		:gamma:
		:res_array:
		:err_array:
		"""
		if self.weight_value: 
			self.weight = np.diag(1/self.cell_area_array)
		#starting gamma_0 and U_array
		gamma_n = dolfinx.fem.Function(self.V0)
		gamma_n.x.array[:] = self.firstguess_array
		U_array = np.array(U_list).flatten()
		U_norm = np.linalg.norm(U_array)

		#starting error and residual lists
		err_array = []
		res_array = []

		#computing step 0 data
		self.set_problem(gamma_n)
		un_list, Un_list = self.solve_problem_current(self.I_all)
		Un_array = np.array(Un_list).flatten()

		### compute step 0 error and residual
		if self.gamma_inv:
			error_form = dolfinx.fem.form(ufl.inner(gamma_n-self.gamma_inv, gamma_n-self.gamma_inv)*ufl.dx)
			err_array.append(dolfinx.fem.assemble_scalar(error_form).real**0.5)
		res_array.append(np.linalg.norm(Un_array - U_array))

		n=0
		while res_array[-1]/U_norm>self.tau*self.noise_level and n<self.step_limit:
			print("starting outer step", n+1)
			### compute step
			gamma_n = self.solve_step(gamma_n, un_list, Un_array, U_array)
			
			### solve direct problem
			self.set_problem(gamma_n)
			un_list, Un_list = self.solve_problem_current(self.I_all)
			Un_array = np.array(Un_list).flatten()

			### compute error and residual
			if self.gamma_inv:
				error_form = dolfinx.fem.form(ufl.inner(gamma_n-self.gamma_inv, gamma_n-self.gamma_inv)*ufl.dx)
				err_array.append(dolfinx.fem.assemble_scalar(error_form).real**0.5)
			res_array.append(np.linalg.norm(Un_array - U_array))
			n += 1

		return (gamma_n,res_array,err_array)

	def solve_step(self,gamma_n,un_list, Un_array,U_array):
		jacobian = self.calc_jacobian(un_list)
		if self.weight_type=="Area":
			adj = self.weight @ jacobian.T.conj()
		elif self.weight_type=="Jacobian":
			jac_norms = np.diag(np.linalg.norm(jacobian,axis=0))
			adj = jac_norms @ self.weight @ jacobian.T.conj()
		bn = U_array - Un_array
		residual_norm_n = np.linalg.norm(bn)
		inside_residual = residual_norm_n
		sn_array = np.zeros(self.K,dtype=PETSc.ScalarType)
		
		k=0

		if self.inner_method == "Landweber":  
			while np.linalg.norm(inside_residual) >= self.mu_i * residual_norm_n and k<self.innerstep_limit:
				inside_residual = jacobian @ sn_array - bn
				sn_array -= self.land_a * adj @ (inside_residual)
				k+=1
			
			print(f"Inner iteration finished with {k} iterations")

			gamma_n_array = gamma_n.x.array + sn_array    
			gamma_n_array.real = np.where(gamma_n_array.real<self.min_v,self.min_v,gamma_n_array.real)
			gamma_n_array.imag = np.where(gamma_n_array.imag<self.min_v,self.min_v,gamma_n_array.imag)
			gamma_n.x.array[:] = gamma_n_array


		elif self.inner_method == "Tikhonov":
			while np.linalg.norm(inside_residual) >= self.mu_i * residual_norm_n and k<self.innerstep_limit:
				inside_residual = jacobian @ sn_array - bn
				alpha_k=self.Tik_c0*(self.Tik_q**k)
				sn_array = np.linalg.solve(adj @ jacobian + alpha_k*np.eye(self.K), adj @bn + alpha_k*sn_array)
				k+=1
			print(f"Inner iteration finished with {k} iterations")
			
			gamma_n_array = gamma_n.x.array + sn_array    
			gamma_n_array.real = np.where(gamma_n_array.real<self.min_v,self.min_v,gamma_n_array.real)
			gamma_n_array.imag = np.where(gamma_n_array.imag<self.min_v,self.min_v,gamma_n_array.imag)
			gamma_n.x.array[:] = gamma_n_array
		
		return gamma_n

	def calc_jacobian(self, un_list):
		"""Calculate the derivative matrix (Jacobian).

		This method calculates the derivative matrix (Jacobian) required for the inverse EIT problem.

		:returns: (ndarray) -- Returns the derivative matrix.
		"""
		y_list, Y_list = self.solve_adjoint_problem(list(self.id_matrix))
		grad_y_list = [compute_gradient(y_func) for y_func in y_list]
		grad_u_list = [compute_gradient(u) for u in un_list]
		l = len(un_list)

		jacobian = np.zeros((l*self.L,self.K),dtype=PETSc.ScalarType)
		for h in range(l): #For each experiment
			for j in range(self.L): #for each electrode
				jacobian[h*self.L+j] = -1*np.sum(grad_u_list[h]*grad_y_list[j].conj(), axis=1)*self.cell_area_array

			 
		return jacobian

