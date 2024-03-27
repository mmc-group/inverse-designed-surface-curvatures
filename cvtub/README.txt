Curvatubes is a Python code for generating tubular, membranous and porous 3D shape textures.

This code is obtained from the original cvtub code by Anna Song, the original theoretical framework is provided in the paper: A. Song, Generation of Tubular and Membranous Shape Textures with Curvature Functionals, J Math Imaging Vis 64, 17–40 (2022). https://doi.org/10.1007/s10851-021-01049-9.

The original code can be found in https://github.com/annasongmaths/curvatubes.

These shapes are modeled as optimizers of a curvature functional F(S) representing a curvature-based geometric energy of surfaces. This functional generalizes the classical Willmore and Helfrich energies, by allowing the principal curvatures to play non-symmetric roles.

The original problem is approximated by a phase-field volumetric energy Feps(u), which transposes the optimization on 2D surfaces S to 3D scalar fields u. This allows for an efficient and flexible GPU algorithm, where topological changes encountered during the flow are addressed seamlessly. The implementation benefits from the automatic differentiation engine provided by PyTorch, combined to optimizers such as Adam and L-BFGS.

Curvatubes leads to a wide continuum of shape textures, encompassing tubules and membranes of all sorts, such as porous anisotropic structures or highly branching networks. In particular, many more patterns than those presented in the publication can be generated.
