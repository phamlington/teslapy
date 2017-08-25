TESLaCU mpi-decomposed object classes for reading, writing, and analyzing data from CFD simulations. Originally written with Athena-RFX in mind, they are nevertheless fully adaptable to any CFD package, especially those packages that use serial and/or parallel IO libraries that have Python wrappers (e.g. HDF5, VTK, and NetCDF all have Python implementations).

this folder includes three classes: mpiAnalyzer, mpiReader, and mpiWriter

currently mpiReader and mpiWriter only provide readers and writers for MPI-IO raw binary

mpiAnalyzer has a fully-functioning factory function and the user can choose between different pure-Python numerical backends (Akima-spline-based flux differencing, central finite differencing, and Fourier differentiation), however I have only ever used this class to analyze HIT from Athena-RFX and so it comes without warranty for any other purpose for the time being
