import dolfinx
import pyvista

def plot_mesh(mesh):
  # Ploting mesh

	p = pyvista.Plotter(notebook=True)
	grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
	p.add_mesh(grid, show_edges=True)
	p.view_xy()
	if pyvista.OFF_SCREEN:
		figure = p.screenshot("disk.png")
	p.show()

def plot_tent_function(u,savefile=False, filename='',plot_imaginary=False):
	# Ploting
	V_u = u.function_space
	grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V_u))
	grid.point_data["Real part"] = u.x.array.real
	grid.point_data["Imag. part"] = u.x.array.imag

	if plot_imaginary:
		p = pyvista.Plotter(shape=(1,2),notebook=True,window_size=(800,400))#,shape=(1,2),)

		p.subplot(0,0)
		p.add_mesh(grid,scalars="Real part",show_edges=True,copy_mesh=True)
		p.view_xy()
		p.subplot(0,1)
		p.add_mesh(grid,scalars="Imag. part",show_edges=True,copy_mesh=True)
	else:
		p = pyvista.Plotter(notebook=True,window_size=(400,400))#,shape=(1,2),)
		p.add_mesh(grid,scalars="",show_edges=True,copy_mesh=True)

	p.view_xy()
	p.set_background("white")

	if not pyvista.OFF_SCREEN:
		p.show(jupyter_backend="static")
	if savefile:
		p.screenshot(filename+".png") 

	p.close()	

def plot_indicator_function(u,savefile=False, filename='',plot_imaginary=False):
	# Ploting
	pyvista.start_xvfb()
	u_mesh = u.function_space.mesh
	grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u_mesh,dim=2))
	# grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)

	grid.cell_data["Real part"] = u.x.array.real
	grid.cell_data["Imag. part"] = u.x.array.imag
	# p.add_text("U real solution", position="upper_edge", font_size=14, color="black")

	if plot_imaginary:
		p = pyvista.Plotter(shape=(1,2),notebook=True,window_size=(800,400))#,shape=(1,2),)

		p.subplot(0,0)
		p.add_mesh(grid,scalars="Real part",show_edges=True,copy_mesh=True)
		p.view_xy()
		p.subplot(0,1)
		p.add_mesh(grid,scalars="Imag. part",show_edges=True,copy_mesh=True)
	else:
		p = pyvista.Plotter(notebook=True,window_size=(400,400))
		p.add_mesh(grid,scalars="Real part",show_edges=True,copy_mesh=True)

	p.view_xy()
	p.set_background("white")
	if not pyvista.OFF_SCREEN:
		p.show(jupyter_backend="static")
	if savefile:
		p.screenshot(filename+".png")    

	p.close()
