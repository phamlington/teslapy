"""
Analysis of compressible, ideal gas, HIT using draft versions of TESLaCU
python modules.

Command Line Options:
---------------------
-i <input directory>    default: 'data/'
-o <output directory>   default: 'analysis/'
-p <problem ID>         defualt: 'no_problem_id'
-N <Nx>                 default: 512
-g <gamma>              default: 1.4
-L <L>                  default: 1.0
-R <R>                  default: 8.3144598e7/21
-r <irs:ire:rint>       default: 1:20:1
-t <its:ite:tint>       default: 1:20:1
--Texp <texp>           default: 0.7
--Tcoef <tcoef>         default: 3.1e-6
--Tmp0 <tmp0>           default: 293.0

Notes:
------

Definitions:
------------

Authors:
--------
Colin Towery, colin.towery@colorado.edu

Turbulence and Energy Systems Laboratory
Department of Mechanical Engineering
University of Colorado Boulder
http://tesla.colorado.edu
"""
from mpi4py import MPI
import sys
import numpy as np
import mpiAnalyzer
import mpiReader
# import mpiWriter
from single_comm_functions import *
from scipy.interpolate import Akima1DInterpolator as interp
comm = MPI.COMM_WORLD


###############################################################################
def ke_enstrophy_budgets(args):
    if comm.rank == 0:
        print("Python MPI job `ke_enstrophy_budgets' started with "
              "{} tasks at {}.".format(comm.size, timeofday()))

    (idir, odir, pid, N, L, irs, ire, rint, its, ite, tint, gamma, R,
     texp, tcoef, T_0) = args

    ir = irs
    # nr = (ire-irs)/rint + 1
    nt = (ite-its)/tint + 1
    # Ne = nr*N**3

    prefix = ['Density', 'Velocity1', 'Velocity2', 'Velocity3',
              'Total_Energy', 'Scalar0',
              'Pressure', 'Reaction', 'Diffusion', 'Momentum', 'Dilatation']

    # -------------------------------------------------------------------------
    # Divide COMM_WORLD amongst the data snapshots

    if N % comm.size > 0:
        if comm.rank == 0:
            print ('Job started with improper number of MPI tasks for the '
                   'size of the data specified!')
        MPI.Finalize()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Generate a data reader and analyzer with the appropriate MPI comms

    reader = mpiReader.mpiBinaryReader(
                mpi_comm=comm, idir=idir, ndims=3,
                decomp=[True, False, False], nx=[N]*3, nh=None,
                periodic=[True]*3, byteswap=False)

    analyzer = mpiAnalyzer.factory(
                comm=comm, idir=idir, odir=odir, probID=pid, L=L, nx=N,
                geo='hit', method='akima')

    update = '{0}\t{1}\t{2}\t{3:4d}\t%s'.format
    fmt = '{0}/{0}_{1}_{2}.bin'.format

    s = list(analyzer.nnx)
    dEdt = np.empty(s, dtype=np.float64)

    s.insert(0, 3)
    u = np.empty(s, dtype=np.float64)

    s = list(analyzer.nnx)
    s.insert(0, nt)
    Enst = np.empty(s, dtype=np.float64)

    analyzer.tol = 1.0e-16

    for it in xrange(its, ite+1, tint):
        tstep = str(it).zfill(4)

        u[0] = reader.get_variable(fmt(prefix[1], tstep, ir))
        u[1] = reader.get_variable(fmt(prefix[2], tstep, ir))
        u[2] = reader.get_variable(fmt(prefix[3], tstep, ir))

        Enst[it] = 0.5*np.sum(np.square(analyzer.curl(u)), axis=0)

    if comm.rank == 0:
        lines = np.loadtxt('%sTimes_%d.txt' % (idir, ir))
        # lines.split
        # grep on it

    ti = np.array([0.5*(t[2]+t[1]), 0.5*(t[3]+t[2])])
    dt_inv = 2.0/(t[3]-t[1])

    for k in xrange(s[1]):
        for j in xrange(s[2]):
            for i in xrange(s[3]):
                spline = interp(t, Enst[:, k, j, i])
                tmp = spline(ti)
                dEdt[k, j, i] = (tmp[1] - tmp[0])

    dEdt *= dt_inv

    if comm.rank % 64 == 0:
        print (update(timeofday(), tstep, ir, comm.rank)
               % 'dEdt computed')

    # ----------------------------------------------------------
    it = (its+ite)/2
    tstep = str(it).zfill(4)

    rho = reader.get_variable(fmt(prefix[0], tstep, ir))
    u[0] = reader.get_variable(fmt(prefix[1], tstep, ir))
    u[1] = reader.get_variable(fmt(prefix[2], tstep, ir))
    u[2] = reader.get_variable(fmt(prefix[3], tstep, ir))
    P = reader.get_variable(fmt(prefix[6], tstep, ir))

    if comm.rank % 64 == 0:
        print (update(timeofday(), tstep, ir, comm.rank)
               % 'variables loaded into memory')

    T = P/(rho*R)
    Smm = analyzer.div(u)
    mu = tcoef*np.power(T, texp)
    omega = analyzer.curl(u)
    Enst = 0.5*np.sum(np.square(analyzer.curl(u)), axis=0)

    A = analyzer.grad(u)
    S = A
    S = 0.5*(A + np.rollaxis(A, 1))

    Adv = np.zeros_like(rho)
    VS = np.zeros_like(rho)
    BCT = np.zeros_like(rho)
    Dif = np.zeros_like(rho)

    for j in range(3):
        for i in range(3):
            Adv += omega[i]*u[j]*analyzer.deriv(omega[i], dim=j)
            VS += omega[i]*omega[j]*S[j, i]

    Dil = 2.0*Enst*Smm

    e = np.zeros((3, 3, 3))
    e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = 1
    e[0, 2, 1] = e[2, 1, 0] = e[1, 0, 2] = -1
    for k in range(3):
        for j in range(3):
            for i in range(3):
                BCT += (e[i, j, k]*omega[i]*analyzer.deriv(rho, dim=j)
                        *analyzer.deriv(P, dim=k))
    BCT *= 1.0/rho**2

    Dif += (1.0/3.0)*np.sum(
            omega*analyzer.curl(analyzer.div(mu*Smm)/rho), axis=0)

    tau = np.zeros_like(u)
    for j in range(3):
        tau[j] = analyzer.div(mu*S[j])

    Dif += 2.0*np.sum(omega*analyzer.curl(analyzer.div(tau)/rho),
                      axis=0)

    LHS = dEdt + Adv
    RHS = VS - Dil + BCT + Dif

    if comm.rank % 64 == 0:
        print (update(timeofday(), tstep, ir, comm.rank)
               % 'variables computed')

    gmin = comm.allreduce(np.min(Enst), op=MPI.MIN)
    gmax = comm.allreduce(np.max(Enst), op=MPI.MAX)
    gmean = comm.allreduce(psum(Enst), op=MPI.SUM)/N**3
    scalar_analysis(analyzer, Enst, (gmin, gmax), gmean,
                    None, None, 'Enst', 'enstrophy', '\Omega')

    gmin = comm.allreduce(np.min(VS), op=MPI.MIN)
    gmax = comm.allreduce(np.max(VS), op=MPI.MAX)
    gmean = comm.allreduce(psum(VS), op=MPI.SUM)/N**3
    scalar_analysis(analyzer, VS, (gmin, gmax), gmean,
                    None, None, 'VS', 'vortex stretching', 'VS')

    gmin = comm.allreduce(np.min(BCT), op=MPI.MIN)
    gmax = comm.allreduce(np.max(BCT), op=MPI.MAX)
    gmean = comm.allreduce(psum(BCT), op=MPI.SUM)/N**3
    scalar_analysis(analyzer, BCT, (gmin, gmax), gmean,
                    None, None, 'BCT', 'baroclinic torque', 'BCT')

    gmin = comm.allreduce(np.min(Dil), op=MPI.MIN)
    gmax = comm.allreduce(np.max(Dil), op=MPI.MAX)
    gmean = comm.allreduce(psum(Dil), op=MPI.SUM)/N**3
    scalar_analysis(analyzer, Dil, (gmin, gmax), gmean,
                    None, None, 'Dil', 'Dilatation', '\Omega\Theta')

    gmin = comm.allreduce(np.min(Dif), op=MPI.MIN)
    gmax = comm.allreduce(np.max(Dif), op=MPI.MAX)
    gmean = comm.allreduce(psum(Dif), op=MPI.SUM)/N**3
    scalar_analysis(analyzer, Dif, (gmin, gmax), gmean,
                    None, None, 'Dif', 'Diffusion', 'Diff')

    gmin = comm.allreduce(np.min(RHS), op=MPI.MIN)
    gmax = comm.allreduce(np.max(RHS), op=MPI.MAX)
    gmean = comm.allreduce(psum(RHS), op=MPI.SUM)/N**3
    scalar_analysis(analyzer, RHS, (gmin, gmax), gmean,
                    None, None, 'RHS', 'RHS', 'RHS')

    gmin = comm.allreduce(np.min(dEdt), op=MPI.MIN)
    gmax = comm.allreduce(np.max(dEdt), op=MPI.MAX)
    gmean = comm.allreduce(psum(dEdt), op=MPI.SUM)/N**3
    scalar_analysis(analyzer, dEdt, (gmin, gmax), gmean,
                    None, None, 'dEdt', 'time derivative',
                    '\partial\Omega/\partial t')

    gmin = comm.allreduce(np.min(Adv), op=MPI.MIN)
    gmax = comm.allreduce(np.max(Adv), op=MPI.MAX)
    gmean = comm.allreduce(psum(Adv), op=MPI.SUM)/N**3
    scalar_analysis(analyzer, Adv, (gmin, gmax), gmean,
                    None, None, 'Adv', 'advection', 'Adv')

    gmin = comm.allreduce(np.min(LHS), op=MPI.MIN)
    gmax = comm.allreduce(np.max(LHS), op=MPI.MAX)
    gmean = comm.allreduce(psum(LHS), op=MPI.SUM)/N**3
    scalar_analysis(analyzer, LHS, (gmin, gmax), gmean,
                    None, None, 'LHS', 'LHS', '\mathrm{D}\Omega/\mathrm{D}t')

    gmin = comm.allreduce(np.min(LHS-RHS), op=MPI.MIN)
    gmax = comm.allreduce(np.max(LHS-RHS), op=MPI.MAX)
    gmean = comm.allreduce(psum(LHS-RHS), op=MPI.SUM)/N**3
    scalar_analysis(analyzer, LHS-RHS, (gmin, gmax), gmean,
                    None, None, 'epsilon', 'remainder', '\epsilon_\Omega')

    if comm.rank == 0:
        print ("Python MPI job `ke_enstrophy_budgets'"
               " finished at "+timeofday())


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: .8e}'.format})
    ke_enstrophy_budgets(get_inputs())
