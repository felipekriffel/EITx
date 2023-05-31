# FENICSX DOCKER INSTALLING AND SETTING

This file explains procedures used to install and set an Docker enviroment for work with Fenicsx.

Docker may be installed on Ubuntu simply using apt with apt install docker.
However, it is installed by default needing root permissions. Following
procedure may be made to grant permissions:
- [https://docs.docker.com/engine/install/linux-postinstall/] 
- [https://docs.docker.com/engine/install/linux-postinstall/]

## Creating Fenicsx Container

Dolfinx/Fenicsx has a docker image available on the online Docker image
repository, that may be installed and run locally through a docker container.
The following code installs the latest stable version:

``` bash
docker pull dolfinx/dolfinx:stable
```

Besides, it has a image configured to run with Jupyter Labs (which may be needed
even with VS code notebooks extension). Further explanations to run with VS Code
are done in next section. This may be installed through:

```bash
docker pull dolfinx/lab:stable
```

Once installed, it may initialize a container with the following command (in
this case, running Jupyter Lab image):

```bash
docker run --name $name -ti -v $(pwd):/home/fenics/shared -p 8080:80 -d dolfinx/lab:stable 
```

About the options:
 * `--name` specifies an name to check and access the container, specified by `$name`;
 * `-ti` creates container and an interactive terminal;
 * `-v` mounts the local file volume into the container, sharing the files between
   the respective folders.
   * **Important**: The `$(pwd)` stands for the path at the current folder in
     terminal. If you want another directory to be shared, inform its path
     instead of `$(pwd)`;
 * `-p` set the port to the container, as its run communicates as a local server;
 * `-d` runs in background. I.e., it keeps the container running even you exit the terminal with `exit`.

Once running this command, it creates a container and keeps running it in
background. Every running container has an `$id` which is needed to access it and
command later. You may check the actual container id with `docker ps`.

To access container terminal, it may be done through the following command (where `$id` should be replaced with the actual container ID):

```bash
docker exec -it $name bash
```

More about the flags and options:
 * https://cursos.alura.com.br/forum/topico-docker-run-flags-t-i-d-restart-p-263536
   [https://cursos.alura.com.br/forum/topico-docker-run-flags-t-i-d-restart-p-263536]
 * https://stackoverflow.com/questions/48368411/what-is-docker-run-it-flag
   [https://stackoverflow.com/questions/48368411/what-is-docker-run-it-flag]

More about running docker sharing local files:
 * https://fenics.readthedocs.io/projects/containers/en/latest/introduction.html#sharing-files-from-the-host-into-the-container
 * https://fenicsproject.discourse.group/t/workflow-of-docker-jupyter-notebook-with-dolfinx/851


   
## SETTING COMPLEX MODE

FENICSx uses an specific Python library for linear algebra operations: the
PETSc. This package has an issue, which is needing two different installations
for real and complex operations, needing switching between them to work properly
in each case. By default, it comes with the real mode installed, needing switch
to complex mode.

However, docker images has prepared settings to work with real and complex
installing, that may be simply switched through terminal in the container.

To do so, run the docker container terminal with:

```bash
docker exec -it $name bash
```

Once inside the container terminal, runs the following command to activate
complex mode:

```bash
source /usr/local/bin/dolfinx-complex-mode
```
To switch back to real mode, runs:

```bash
source /usr/local/bin/dolfinx-real-mode
```

One can also run this command without opening a terminal in container through:

```bash
docker exec -it $name bash -c "source /usr/local/bin/dolfinx-real-mode"
```

With the mode set, you may simply run any codes and it will be already working properly with the needed case.


## RUNNING JUPYTER NOTEBOOKS WITH DOCKER CONTAINER SERVER

If you run the image dolfinx/lab:stable to start the docker container, it automatically starts and runs a Jupyter Server on it, which is needed to run Notebooks both through Jupyter Notebooks or VSCode extension. To access this server, we need to check the actual server port and generated token, which may be done with the following command:

```bash
docker exec -it $id sh -c "jupyter server list"
```

This outputs the server running inside the container in the format `http://$id:8888/?token=$labtoken`, where `$id` is the container id and `$labtoken` is an access token generated in that server. For example, in my run the output
looks like this:

```bash
[JupyterServerListApp] Currently running servers:
[JupyterServerListApp] http://4d2546eb8a31:8888/?token=b8535df5a6e898c5920e5bc0b315947f26a6e57526cba93d :: /root
```

The URL given in the output second line is the link to the running Jupyter Server. However, in the shown format with the $id, it can not be accessed in the local machine, only through the container. To be able to access it by the local machine, we need first to know the container IP.

You may check the container IP with the following command (where $id stands for the running container id):

```bash
docker inspect --format '{{ .NetworkSettings.IPAddress }}' $id
```

In my case, it outputs:

172.17.0.2

The returned `$ip` is the one that you may access through your local machine. So,
you may access a Jupyter Lab switching the `$id` shown in the Jupyter Server URL
by the container `$ip` obtained now. For example, in my case it becomes:

```
https://172.17.0.2:8888/?token=b8535df5a6e898c5920e5bc0b315947f26a6e57526cba93d
```

Accessing this link through a browser opens a Jupyter Lab interface that will be
ready to run.
 * Obs: if the returned IP does not works when trying to accessing it, check the
   following discussion for alternatives: https://stackoverflow.com/questions/27191815/how-to-access-docker-containers-web-server-from-host


Notice that Jupyter Lab interface shows two kernels as options to start a notebook or console: the default iPython kernel and the Python 3 (DOLFINx) kernel. Chose the DOLFINx option to run it in with the right settings.

To run it on VSCode, opening a .ipynb file, open the VSCode command pallet (F1 shortcut), run the command

```
Notebook: Select Notebook Kernel > Existing Jupyter Server
```

Then, inform the URL of Jupyter Server (with the container IP instead of ID). Once connected, select the option Python 3 (DOLFINx complex) in the dropdown.

With this, you should be able to go with your FENICSx codes in VSCode Jupyter Notebooks :).