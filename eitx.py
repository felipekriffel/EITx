import sys
import dolfinx
from mpi4py import MPI #import parallel communicator
import numpy as np
import ufl
import matplotlib.pyplot as plt
import dolfinx.fem.petsc
from dolfinx.io import gmshio
import gmsh
import pyvista
import time
import pandas as pd
import scipy as sp
import basix
from petsc4py import PETSc #Linear algebra lib


class Electrodes():
    """
    Object that contains the position of the electrodes in the boundary.
    The position is stored as the initial and final angles of each electrode.

    :param L: Number of electrodes.
    :type L: int
    :param per_cober: Percentual covered length in the boundary by electrodes, between 0 and 1.
    :type per_cober: float
    :param rotate: Rotation angle in the original solution for electrodes (in radians).
    :type rotate: float
    :param anticlockwise: If True, the electrodes are positioned anticlockwise, else clockwise.
    :type anticlockwise: bool, optional (default is True)
    """

    def __init__(self, L, per_cober, rotate, anticlockwise=True):
        #Checks
        if not isinstance(L, int): raise ValueError("Number of electrodes must be an integer.")
        if not isinstance(per_cober, float): raise ValueError("per_cober must be a float.")
        if not isinstance(rotate, (int, float)): raise ValueError("rotate must be a float.")
        if not isinstance(anticlockwise, bool): raise ValueError("anticlockwise must be true of false.")
        if per_cober>1: raise ValueError("per_cober must be equal or less than 1. Example (75%): per_cober=0.75 ")

        self.rotate=rotate
        self.L=L
        self.per_cober=per_cober
        self.anticlockwise=anticlockwise

        self.position=self.calc_position()
        self.coordinates = [((np.cos(electrode[0]),np.sin(electrode[0])), (np.cos(electrode[1]),np.sin(electrode[1]))) for electrode in self.position]


    def calc_position(self):
        """
        Calculate the position of electrodes based on the :class:`electrodes_position` object.

        :returns: list of arrays -- Returns a list with angle initial and final of each electrode.
        """
        size_e=2*np.pi/self.L*self.per_cober       #Size electrodes
        size_gap=2*np.pi/self.L*(1-self.per_cober) #Size gaps
        rotate=self.rotate                      #Rotating original solution

        electrodes=[]
        for i in range(self.L):
            #Example electrodes=[[0, pi/4], [pi/2, pi]]
            electrodes.append([size_e*i+size_gap*i+rotate, size_e*(i+1)+size_gap*i+rotate]) #Grouping angular values for electrodes.
        if not self.anticlockwise:
            electrodes[1:] = electrodes[1:][::-1] #Keep first electrode and reverse order
        return electrodes


class MeshClass:
  def __init__(self,electrodes, mesh_refining=1,bdr_refining=1):
    self.electrodes = electrodes
    self.mesh, self.cell_markers, self.facet_markers = self.setup_mesh(mesh_refining,bdr_refining)
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
        facets = dolfinx.mesh.locate_entities(self.mesh, self.mesh.topology.dim - 1, locator)
        bdr_facet_indices.append(facets)
        bdr_facet_markers.append(np.full_like(facets, marker))
    bdr_facet_indices = np.hstack(bdr_facet_indices).astype(np.int32)
    bdr_facet_markers = np.hstack(bdr_facet_markers).astype(np.int32)
    sorted_facets = np.argsort(bdr_facet_indices)
    facet_tag = dolfinx.mesh.meshtags(self.mesh, self.mesh.topology.dim - 1, bdr_facet_indices[sorted_facets], bdr_facet_markers[sorted_facets])

    return ufl.Measure("ds", domain=self.mesh, subdomain_data=facet_tag)

class DirectProblem:
  def __init__(self, mesh_object: MeshClass, z_values):
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

    self.set_submatrices()

  def set_submatrices(self):
    v = ufl.TestFunction(self.V)
    ###
    #A2
    A2_form_array = [-ufl.inner((1/self.z_petsc_array[i]), v) * self.ds(i) for i in range(self.L)]
    A2_form_obj = [dolfinx.fem.form(form) for form in A2_form_array]
    A2_vector_array= [dolfinx.fem.petsc.assemble_vector(form) for form in A2_form_obj]
    self.A2_np_matrix = np.stack(
      [A2_vector_array[j].getValues(range(self.N)) for j in range(self.L)],
      axis=1
    )

    ###
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


  def solve_problem_current(self, I_all,save=False):
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
    if not self.assembled:
      raise Exception("The problem operator has not been assembled")

    u_list = []
    u_array_list = []
    U_list = []

    l = len(I_all)

    for k in range(l):
      b = np.block([np.zeros(self.N),I_all[k]])
      u_nest = sp.sparse.linalg.spsolve(self.A_op,self.A_np.T.conj()@b) #Solve A^*Au = b
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
    gradw = ufl.grad(w)
    gradv = ufl.grad(v)
    for us in u_list:
      gradus = ufl.grad(us)

      #creating form
      L_exp = - (eta * ufl.inner(gradus,gradv)) * ufl.dx
      L_form = dolfinx.fem.form(L_exp)

      #getting rhs vector
      L_vector = dolfinx.fem.petsc.assemble_vector(L_form)
      L_nparray = L_vector.getValues(range(L_vector.getSize()))

      L_array.append(np.block([L_nparray,np.zeros(L)]))


    ws, Ws = self.solve_problem_vector(L_array)
    return ws,Ws

  def adjoint(self, sigma_list, u_list):
    """
    Get sigma_list direction, returns F'(gamma)* sigma_list
    """
    if not self.adjoint_set:
      self.setAdjointProblem()

    psi_list, Psi_list  = self.problem_adj.solve_problem_current(sigma_list)
    grad_psi_list = [compute_gradient(psi) for psi in psi_list]
    grad_u_list = [compute_gradient(u) for u in u_list]
    adj = dolfinx.fem.Function(self.V0)
    adj_j_array = np.zeros_like(adj.x.array)

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
      u_nest = sp.sparse.linalg.spsolve(self.A_op,self.A_np.T.conj()@b) #Solve A^*Au = b
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
    

class InverseProblem(DirectProblem):
  def __init__(self,mesh_inverse, z_values):
    super().__init__(mesh_inverse,z_values)
  
    #"First guess and weight functions"
    self.firstguess=np.ones(self.K)           #First guess for Forwardproblem
    self.weight=np.ones(self.K)             #Initial weight function
    
    #"Solver configurations"
    self.verbose=False
    self.weight_value=False    #Are you going to use the weight function in the Jacobian matrix?
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
    self.gamma0=None  #Exact Solution
    self.mesh0=None   #Mesh of exact solution
    
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
    pass

  def solve_innerNewton(self, Jacobiana_all,b0):
    """Solve the inner step of the Newton method.

    This method is used to solve the inner step of the Newton method. It iteratively updates the solution `gamma_k` using different regularization methods.

    :param Jacobiana_all: Derivative Matrix generated by :func:`calc_jacobian()`.
    :type Jacobiana_all: ndarray
    :param b0: Vector containing the difference between the measured potentials and the potentials calculated with the current guess.
    :type b0: array

    :returns: (Array, int, float) -- Returns `sk` (the result of the inner step to add to `gamma_k`), `inner_step` (number of inner steps taken), `mu` (regularization parameter used in the method).
    """        
    #If weight True and step=0, determine the weight.
    
    pass


  def calc_jacobian(self):
    """Calculate the derivative matrix (Jacobian).

    This method calculates the derivative matrix (Jacobian) required for the inverse EIT problem.

    :returns: (ndarray) -- Returns the derivative matrix.
    """
    pass


def plot_mesh(mesh):
  # Ploting mesh

  p = pyvista.Plotter(notebook=True)
  grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
  p.add_mesh(grid, show_edges=True)
  p.view_xy()
  if pyvista.OFF_SCREEN:
      figure = p.screenshot("disk.png")
  p.show()

# gets theta of (x,y) in polar coord. (r,theta), with theta in interval [0,2pi]
def theta(x):
  r = (x[0]**2 + x[1]**2)**(0.5)
  inv_r = np.where(np.isclose(r,0),0,1/r)
  return np.where(x[1]>=0,np.arccos(x[0]*inv_r),2*np.pi - np.arccos(x[0]*inv_r))


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


def plot_tent_function(u):
  pyvista.start_xvfb()
  
  # Ploting
  V_u = u.function_space
  grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V_u))
  grid.point_data["Real part"] = u.x.array.real
  grid.point_data["Imag. part"] = u.x.array.imag

  p = pyvista.Plotter(shape=(1,2),notebook=True,window_size=(800,400))#,shape=(1,2),)

  p.subplot(0,0)
  p.add_mesh(grid,scalars="Real part",show_edges=True,copy_mesh=True)
  p.view_xy()
  p.subplot(0,1)
  p.add_mesh(grid,scalars="Imag. part",show_edges=True,copy_mesh=True)
  p.view_xy()
  p.set_background("white")
  if not pyvista.OFF_SCREEN:
      p.show(jupyter_backend="static")

def plot_indicator_function(u):
  # Ploting
  pyvista.start_xvfb()
  u_mesh = u.function_space.mesh
  grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u_mesh,dim=2))
  # grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)

  grid.cell_data["Real part"] = u.x.array.real
  grid.cell_data["Imag. part"] = u.x.array.imag
  # p.add_text("U real solution", position="upper_edge", font_size=14, color="black")
  p = pyvista.Plotter(shape=(1,2),notebook=True,window_size=(800,400))#,shape=(1,2),)

  p.subplot(0,0)
  p.add_mesh(grid,scalars="Real part",show_edges=True,copy_mesh=True)
  p.view_xy()
  p.subplot(0,1)
  p.add_mesh(grid,scalars="Imag. part",show_edges=True,copy_mesh=True)
  p.view_xy()
  p.set_background("white")
  if not pyvista.OFF_SCREEN:
      p.show(jupyter_backend="static")


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

def funcProduct(u,v):
  product = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(u,v) * ufl.dx))
  return product

def funcSquareNorm(u):
  return dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(u,u) * ufl.dx))
