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
import numpy as np
import mpiAnalyzer
import mpiReader
from single_comm_functions import *
import sys
comm = MPI.COMM_WORLD


###############################################################################
def autoignition_statistical_analysis(args):
    if comm.rank == 0:
        print("Python MPI job `autoignition_statistical_analysis' "
              "started with {} tasks at {}.".format(comm.size, timeofday()))

    (idir, odir, pid, N, L, irs, ire, rint, its, ite, tint, gamma, R,
     texp, tcoef, T_0) = args

    nr = (ire-irs)/rint + 1
    nt = (ite-its)/tint + 1
    Ne = nr*N**3

    prefix = ['Density', 'Velocity1', 'Velocity2', 'Velocity3',
              'Total_Energy', 'Scalar0', 'Velocity_Perturbations1',
              'Velocity_Perturbations2', 'Velocity_Perturbations3']

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

    update = '{0}\t{1}\t{2}\t{3:4d}\t'.format
    fmt = '{0}/{0}_{1}_{2}.bin'.format
    Ek_fmt = "\widehat{{{0}}}^*\widehat{{{0}}}".format

    s = list(analyzer.nnx)
    s.insert(0, 3)
    u = np.empty(s, dtype=np.float64)
    analyzer.tol = 1.0e-16

    psum = mpiAnalyzer.psum
    MMIN = MPI.MIN
    MMAX = MPI.MAX
    MSUM = MPI.SUM

    B = 1.0e13
    T_a = 13500

    for it in xrange(its, ite+1, tint):
        tstep = str(it).zfill(4)

        emin = np.empty(50)
        emax = np.empty(50)

        emin[:] = np.inf
        emax[:] = np.NINF

        for ir in xrange(irs, ire+1, rint):

            rho = reader.get_variable(fmt(prefix[0], tstep, ir))
            u[0] = reader.get_variable(fmt(prefix[1], tstep, ir))
            u[1] = reader.get_variable(fmt(prefix[2], tstep, ir))
            u[2] = reader.get_variable(fmt(prefix[3], tstep, ir))
            re = reader.get_variable(fmt(prefix[4], tstep, ir))
            rY = reader.get_variable(fmt(prefix[5], tstep, ir))

            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'variables loaded into memory')

            # -------------------------------------------------------------------------

            ek = 0.5*np.sum(np.square(u), axis=0)   # specific kinetic energy
            rei = re-rho*ek                         # internal energy density
            P = rei*(gamma-1)                       # thermodynamic pressure
            T = P/(rho*R)                           # thermodynamic temperature
            Y = rY/rho
            rwdot = -rho*rY*B*np.exp(-T_a/T)
            # mu = tcoef*np.power(T, texp)
            Smm = analyzer.div(u)

            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'variables computed')

            iv = 0

            emin[iv] = min(emin[iv], comm.allreduce(np.min(rho), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(rho), op=MMAX))
            iv+=1

            # if comm.rank % 64 == 0:
            #     print (update(timeofday(), tstep, ir, comm.rank)
            #            +'rho min max mean computed')

            vm = np.sum(np.square(u), axis=0)
            emin[iv] = min(emin[iv], comm.allreduce(np.min(vm), op=MPI.MIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(vm), op=MPI.MAX))
            iv+=1

            emin[iv] = min(emin[iv], comm.allreduce(np.min(P), op=MPI.MIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(P), op=MPI.MAX))
            iv+=1

            emin[iv] = min(emin[iv], comm.allreduce(np.min(T), op=MPI.MIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(T), op=MPI.MAX))
            iv+=1

            c_s = vm = np.sqrt(gamma*R*T)
            # emin[iv] = min(emin[iv], comm.allreduce(np.min(c_s), op=MPI.MIN))
            # emax[iv] = max(emax[iv], comm.allreduce(np.max(c_s), op=MPI.MAX))
            # iv+=1

            M = vm = np.sqrt(np.sum(np.square(u), axis=0))/c_s
            emin[iv] = min(emin[iv], comm.allreduce(np.min(M), op=MPI.MIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(M), op=MPI.MAX))
            iv+=1

            emin[iv] = min(emin[iv], comm.allreduce(np.min(Smm), op=MPI.MIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(Smm), op=MPI.MAX))
            iv+=1

            emin[iv] = min(emin[iv], comm.allreduce(np.min(rY), op=MPI.MIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(rY), op=MPI.MAX))
            iv+=1

            emin[iv] = min(emin[iv], comm.allreduce(np.min(Y), op=MPI.MIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(Y), op=MPI.MAX))
            iv+=1

            emin[iv] = min(emin[iv], comm.allreduce(np.min(rwdot), op=MPI.MIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(rwdot), op=MPI.MAX))

            comm.Barrier()
            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'min/max loop finished')

            # -----------------------------------------------------------------

        rhom = comm.allreduce(psum(rho), op=MSUM)/Ne

        # ---------------------------------------------------------------------
        # BEGIN ANALYSIS

        for ir in xrange(irs, ire+1, rint):
            analyzer.mpi_moments_file = '%s%s-%s_%s.moments' % (
                                        analyzer.odir, pid, tstep, str(ir))
            analyzer.prefix = '%s-%s_%s_' % (pid, tstep, str(ir))
            analyzer.Nx = Ne
            if comm.rank == 0:
                try:
                    os.remove(analyzer.mpi_moments_file)
                except:
                    pass

            rho = reader.get_variable(fmt(prefix[0], tstep, ir))
            u[0] = reader.get_variable(fmt(prefix[1], tstep, ir))
            u[1] = reader.get_variable(fmt(prefix[2], tstep, ir))
            u[2] = reader.get_variable(fmt(prefix[3], tstep, ir))
            re = reader.get_variable(fmt(prefix[4], tstep, ir))
            rY = reader.get_variable(fmt(prefix[5], tstep, ir))

            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'variables loaded into memory')

            # -------------------------------------------------------------------------

            ek = 0.5*np.sum(np.square(u), axis=0)   # specific kinetic energy
            rei = re-rho*ek                         # internal energy density
            P = rei*(gamma-1)                       # thermodynamic pressure
            T = P/(rho*R)                           # thermodynamic temperature
            Y = rY/rho
            rwdot = -rho*rY*B*np.exp(-T_a/T)
            # mu = tcoef*np.power(T, texp)
            Smm = analyzer.div(u)
            Enst = 0.5*np.sum(np.square(analyzer.curl(u)), axis=0)

            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'variables computed')

            iv = 0

            scalar_analysis(analyzer, rho, (emin[iv], emax[iv]), None, None,
                            'rho', 'density', '\\rho')
            iv+=1

            analyzer.spectral_density(u, 'u', 'velocity PSD', Ek_fmt('u_i'))

            vm = np.sum(np.square(u), axis=0)
            scalar_analysis(analyzer, vm, (emin[iv], emax[iv]), None, None,
                            'uiui', 'velocity squared', 'u_iu_i')
            scalar_analysis(analyzer, vm, (emin[iv], emax[iv]), rho, rhom,
                            'uiui_tilde', 'm.w. velocity squared', 'u_iu_i')
            iv+=1

            scalar_analysis(analyzer, P, (emin[iv], emax[iv]), None, None,
                            'P', 'pressure', 'P')
            iv+=1

            comm.Barrier()
            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'\tvelocity analyses completed')
            # -----------------------------------------------------------------

            scalar_analysis(analyzer, T, (emin[iv], emax[iv]), None, None,
                            'T', 'temperature', 'T')
            scalar_analysis(analyzer, T, (emin[iv], emax[iv]), rho, rhom,
                            'T_tilde', 'm.w. temperature', 'T')
            iv+=1

            c_s = vm = np.sqrt(gamma*R*T)
            analyzer.write_mpi_moments(c_s, 'speed of sound', 'c_\\mathrm{s}',
                                       None, None, m1=0)
            analyzer.write_mpi_moments(c_s, 'm.w. speed of sound',
                                       'c_\\mathrm{s}', rho, rhom, m1=0)

            M = vm = np.sqrt(np.sum(np.square(u), axis=0))/c_s
            scalar_analysis(analyzer, M, (emin[iv], emax[iv]), None, None,
                            'M', 'Mach number', '\mathrm{Ma}')
            scalar_analysis(analyzer, M, (emin[iv], emax[iv]), rho, rhom,
                            'M_tilde', 'm.w. Mach number', '\mathrm{Ma}')
            iv+=1

            comm.Barrier()
            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'\tthermodynamic analyses completed')
            # -----------------------------------------------------------------

            scalar_analysis(analyzer, Smm, (emin[iv], emax[iv]), None, None,
                            'Smm', 'dilatation', '\Theta')
            scalar_analysis(analyzer, Smm, (emin[iv], emax[iv]), rho, rhom,
                            'Smm_tilde', 'm.w. dilatation', '\Theta')
            iv+=1

            scalar_analysis(analyzer, rY, (emin[iv], emax[iv]), None, None,
                            'rY', 'fuel density', '\\rho Y')
            iv+=1

            scalar_analysis(analyzer, Y, (emin[iv], emax[iv]), None, None,
                            'Y', 'fuel mass-fraction', 'Y')
            scalar_analysis(analyzer, Y, (emin[iv], emax[iv]), rho, rhom,
                            'Y_tilde', 'm.w. fuel mass-fraction', 'Y')
            iv+=1

            scalar_analysis(analyzer, rwdot, (emin[iv], emax[iv]), None, None,
                            'rwdot', 'mass reaction rate', '\\rho\dot{w}')

            # minY = np.min(Y)
            # maxY = np.max(Y)

            # if minY < 0.1 and maxY > 0.7:
            #     # ------------------------------------------------------------

            #     print (update(timeofday(), tstep, ir, comm.rank)
            #            +'found an autoignition kernel!')

            #     with open(analyzer.odir+analyzer.probID+'-slices_{}_{}_{}.bin'
            #               .format(tstep, ir, comm.rank), 'w') as fh:
            #         rho[0].tofile(fh)
            #         P[0].tofile(fh)
            #         T[0].tofile(fh)
            #         Smm[0].tofile(fh)
            #         Enst[0].tofile(fh)
            #         rY[0].tofile(fh)
            #         rwdot[0].tofile(fh)

    if comm.rank == 0:
        print ("Python MPI job `autoignition_statistical_analysis'"
               " finished at "+timeofday())


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: .8e}'.format})
    autoignition_statistical_analysis(get_inputs())
