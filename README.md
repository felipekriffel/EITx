# EITx

Electrical Impedance Tomography library for Python, based on [Fenicsx](https://fenicsproject.org/download/), with support to complex values. It is a Finite Elements Method based implementation.


Some references you may like:

- [Implementação computacional da tomografia por impedância elétrica.](https://coloquio34.impa.br/pdf/34CBM07-eBook.pdf) - Course presented at the 34th Colóquio Brasileiro de Matemática (IMPA - Rio de Janeiro). This describes the theory that based all this implementation, specially the Finite Element Method applied to solve the EIT direct problem in both studied models: Continuous Model and Complete Electrode Model. The appendix also presents an introduction to inverse problems and regularization methods, which are also present in this library;
- [The FEniCSx tutorial](https://jsdokken.com/dolfinx-tutorial/) - Official tutorial of the FEniCSx library, which we used for the FEM utils, assemblers and solver functions;
- [Impedance Tomography for monitoring multiphase flows](https://fabiomargotti.paginas.ufsc.br/prototipo2/) - Project at Federal University of Santa Catarina (Florianópolis - Brasil), coordinated by professor Fábio Margotti, focused on a specific application of the EIT. Particularly, it is the project that gave birth for this library :). Check some of the work done there;


Projects where this is used:
- [EITCNN](https://github.com/felipekriffel/EITCNN) - System developed for some EIT Inverse Problem solving methods, allying Neural Networks to classical mathematical strategies.


Felipe Kaminsky Riffel, 2026