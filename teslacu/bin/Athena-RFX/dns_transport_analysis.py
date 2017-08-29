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
import mpi_single_comm_Analyzer
import mpiReader
from single_comm_functions import *
import sys
import os
comm = MPI.COMM_WORLD


###############################################################################
def physical_transport_analysis(args):
    if comm.rank == 0:
        print("Python MPI job `physical_transport_analysis' started with "
              "{} tasks at {}.".format(comm.size, timeofday()))

    (idir, odir, pid, N, L, irs, ire, rint, its, ite, tint, gamma, R,
     texp, tcoef, tmp0) = args

    nr = (ire-irs)/rint + 1
    nt = (ite-its)/tint + 1
    Ne = nr*nx**3
    prefix = ['Density', 'Velocity1', 'Velocity2', 'Velocity3',
              'Total_Energy', 'Scalar0', 'Velocity_Perturbations1',
              'Velocity_Perturbations2', 'Velocity_Perturbations3']

    # -------------------------------------------------------------------------
    # Divide COMM_WORLD amongst the data snapshots
    # ENS_COMM is the communicator for purely ENSemble statistics (all spatial
    # data over a single timestep)
    # SP_COMM  is the communicator for a single snapshot of SPatial data
    # ROOT_COMM is the (intra)communicator between root tasks of each SP_COMM
    # in an ENS_COMM

    if nx % comm.size > 0:
        if comm.rank == 0:
            print ('Job started with improper number of MPI tasks for the '
                   'size of the data specified!')
        MPI.Finalize()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Generate a data reader and analyzer with the appropriate MPI comms

    reader = mpiReader.mpiBinaryReader(
                mpi_comm=comm, idir=idir, ndims=3,
                decomp=[True, False, False], nx=[nx]*3, nh=None,
                periodic=[True]*3, byteswap=False)

    analyzer = mpi_single_comm_Analyzer.factory(
                comm=comm, idir=idir, odir=odir, probID=pid, L=L, nx=nx,
                geo='hit', method='akima')

    update = '{0}\t{1}\t{2}\t{3:4d}\t'.format
    fmt = '{}_{}_{}.bin'.format

    s = list(analyzer.nnx)
    s.insert(0, 3)
    u = np.empty(s, dtype=np.float64)
    v = np.empty_like(u)
    analyzer.tol = 1.0e-16

    psum = analyzer.psum
    MMIN = MPI.MIN
    MMAX = MPI.MAX
    MSUM = MPI.SUM

    ddx = '\\frac{{\partial}}{{\partial x_j}}'

    strings = {
        A1: 'u_i%s(\\rho u_iu_j)' % ddx,
        A2: '\\rho u_iu_j%s(u_i)' % ddx,
        divKu: '%s(\\rho u_iu_iu_j)' % ddx,
        Ares: '\epsilon^\mathrm{{res}}_\mathrm{{A}}',
        P1: 'u_j%s(P)' % ddx,
        Pdil: 'P\Theta',
        divPu: '%s(Pu_j)' % ddx,
        Pres: '\epsilon^\mathrm{{res}}_\mathrm{{P}}',
        D1: 'u_i%s(\sigma_{ij})' % ddx,
        D2: '\sigma_{ij}S_{ij}',
        divSu: '%s(\sigma_{ji}u_i)' % ddx,
        Dres: '\epsilon^\mathrm{{res}}_\mathrm{{D}}'}

    # -------------------------------------------------------------------------
    # Start first set of loops to determine ensemble and temporal mins, maxs,
    # and means

    # tmin = np.zeros(31)
    # tmax = np.zeros(31)
    # tmean= np.zeros(53)

    for it in xrange(its, ite+1, tint):
        tstep = str(it).zfill(4)

        nv = 16
        emin = np.empty(nv)
        emax = np.empty(nv)
        emean= np.zeros(nv)

        emin[:] = np.inf
        emax[:] = np.NINF

        for ir in xrange(irs, ire+1, rint):
            analyzer.prefix = '%s-%s_%s_' % (pid, tstep, str(ir))
            analyzer.mpi_moments_file = ('%s%s%s.moments' % (analyzer.odir,
                                         analyzer.prefix, 'all_stats'))
            if comm.rank == 0:
                try:
                    os.remove(analyzer.mpi_moments_file)
                except:
                    pass

            rho = reader.variable_subdomain(fmt(prefix[0], tstep, ir))
            u[0] = reader.variable_subdomain(fmt(prefix[1], tstep, ir))
            u[1] = reader.variable_subdomain(fmt(prefix[2], tstep, ir))
            u[2] = reader.variable_subdomain(fmt(prefix[3], tstep, ir))
            re = reader.variable_subdomain(fmt(prefix[4], tstep, ir))

            iv = 0

            P = re = (re-rho*0.5*np.sum(np.square(u), axis=0))*(gamma-1)
            mu = tcoef*np.power(P/(rho*R), texp)

            A = analyzer.grad(u)
            e = np.zeros((3, 3, 3))
            e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = 1
            e[0, 2, 1] = e[2, 1, 0] = e[1, 0, 2] = -1
            omga = np.einsum('ijk,jk...->i...', e, A)

            divKu = analyzer.div(0.5*rho*np.sum(np.square(u), axis=0)*u)
            RHS = -divKu
            emin[iv] = min(emin[iv], comm.allreduce(np.min(divKu), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(divKu), op=MMAX))
            emean[iv]+= comm.allreduce(psum(divKu), op=MSUM)
            iv+=1

            S = A
            S = 0.5*(A + np.rollaxis(A, 1))
            Smm = np.einsum('mm...', S)

            Pdil = P*Smm
            RHS += Pdil
            emin[iv] = min(emin[iv], comm.allreduce(np.min(Pdil), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(Pdil), op=MMAX))
            emean[iv]+= comm.allreduce(psum(Pdil), op=MSUM)
            iv+=1

            divPu = divKu = analyzer.div(P*u)
            RHS -= divPu
            emin[iv] = min(emin[iv], comm.allreduce(np.min(divPu), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(divPu), op=MMAX))
            emean[iv]+= comm.allreduce(psum(divPu), op=MSUM)
            iv+=1

            D2 = 2.0*mu*np.sum(np.square(S), axis=(0, 1))
            D2 += mu*np.square(Smm)
            RHS -= D2
            emin[iv] = min(emin[iv], comm.allreduce(np.min(D2), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(D2), op=MMAX))
            emean[iv]+= comm.allreduce(psum(D2), op=MSUM)
            iv+=1

            divSu = divKu = analyzer.div(mu*(np.sum(S*u, axis=0)+u*Smm))
            RHS += divSu
            emin[iv] = min(emin[iv], comm.allreduce(np.min(divSu), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(divSu), op=MMAX))
            emean[iv]+= comm.allreduce(psum(divSu), op=MSUM)
            iv+=1

            # RHS
            emin[iv] = min(emin[iv], comm.allreduce(np.min(RHS), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(RHS), op=MMAX))
            emean[iv]+= comm.allreduce(psum(RHS), op=MSUM)

            vm = mu*np.sum(np.square(omga), axis=0)
            emin[iv] = min(emin[iv], comm.allreduce(np.min(vm), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(vm), op=MMAX))
            emean[iv]+= comm.allreduce(psum(vm), op=MSUM)
            iv+=1


            comm.Barrier()
            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'\t\tmin, max, mean loop finished')

            # -----------------------------------------------------------------

        emean /= Ne

        # if comm.rank == 0:
        #     iv = 0

        #     fh = open(analyzer.mpi_moments_file, 'a')
        #     fh.write(('{:s}\t{:s}\t%s\n' % '\t'.join(['{:.8e}']*3))
        #              .format('Enstrophic dissipation', '\mu\omega_i\omega_i',
        #                      emean[iv], emin[iv], emax[iv]))
        #     iv+=1
        #     fh.write(('{:s}\t{:s}\t%s\n' % '\t'.join(['{:.8e}']*3))
        #              .format('A1', A1str,
        #                      emean[iv], emin[iv], emax[iv]))
        #     iv+=1
        #     fh.write(('{:s}\t{:s}\t%s\n' % '\t'.join(['{:.8e}']*3))
        #              .format('A2', A2str,
        #                      emean[iv], emin[iv], emax[iv]))
        #     iv+=1
        #     fh.write(('{:s}\t{:s}\t%s\n' % '\t'.join(['{:.8e}']*3))
        #              .format('advective flux', Astr,
        #                      emean[iv], emin[iv], emax[iv]))
        #     iv+=1
        #     fh.write(('{:s}\t{:s}\t%s\n' % '\t'.join(['{:.8e}']*3))
        #              .format('residual of advective flux', Ares,
        #                      emean[iv], emin[iv], emax[iv]))
        #     iv+=1
        #     fh.write(('{:s}\t{:s}\t%s\n' % '\t'.join(['{:.8e}']*3))
        #              .format('P1', P1str,
        #                      emean[iv], emin[iv], emax[iv]))
        #     iv+=1
        #     fh.write(('{:s}\t{:s}\t%s\n' % '\t'.join(['{:.8e}']*3))
        #              .format('Pdil', Pdilstr,
        #                      emean[iv], emin[iv], emax[iv]))
        #     iv+=1
        #     fh.write(('{:s}\t{:s}\t%s\n' % '\t'.join(['{:.8e}']*3))
        #              .format('pressure flux', Pstr,
        #                      emean[iv], emin[iv], emax[iv]))
        #     iv+=1
        #     fh.write(('{:s}\t{:s}\t%s\n' % '\t'.join(['{:.8e}']*3))
        #              .format('residual of pressure flux', Pres,
        #                      emean[iv], emin[iv], emax[iv]))
        #     iv+=1
        #     fh.write(('{:s}\t{:s}\t%s\n' % '\t'.join(['{:.8e}']*3))
        #              .format('D1', D1str,
        #                      emean[iv], emin[iv], emax[iv]))
        #     iv+=1
        #     fh.write(('{:s}\t{:s}\t%s\n' % '\t'.join(['{:.8e}']*3))
        #              .format('D2', D2str,
        #                      emean[iv], emin[iv], emax[iv]))
        #     iv+=1
        #     fh.write(('{:s}\t{:s}\t%s\n' % '\t'.join(['{:.8e}']*3))
        #              .format('diffusive flux', Dstr,
        #                      emean[iv], emin[iv], emax[iv]))
        #     iv+=1
        #     fh.write(('{:s}\t{:s}\t%s\n' % '\t'.join(['{:.8e}']*3))
        #              .format('residual of diffusive flux', Dres,
        #                      emean[iv], emin[iv], emax[iv]))
        #     iv+=1
        #     fh.write(('{:s}\t{:s}\t%s\n' % '\t'.join(['{:.8e}']*3))
        #              .format('RHS of kinetic energy budget', '\mathrm{{RHS}}',
        #                      emean[iv], emin[iv], emax[iv]))

        # for i in xrange(29):
        #     tmin[i] = min(tmin[i], emin[i])
        #     tmax[i] = max(tmax[i], emax[i])
        # tmean += emean

        # ---------------------------------------------------------------------
        # BEGIN ANALYSIS

        for ir in xrange(irs, ire+1, rint):

            rho = reader.variable_subdomain(fmt(prefix[0], tstep, ir))
            u[0] = reader.variable_subdomain(fmt(prefix[1], tstep, ir))
            u[1] = reader.variable_subdomain(fmt(prefix[2], tstep, ir))
            u[2] = reader.variable_subdomain(fmt(prefix[3], tstep, ir))
            re = reader.variable_subdomain(fmt(prefix[4], tstep, ir))

            iv = 0

            P = re = (re-rho*0.5*np.sum(np.square(u), axis=0))*(gamma-1)
            mu = tcoef*np.power(P/(rho*R), texp)
            S = analyzer.grad(u)
            e = np.zeros((3, 3, 3))
            e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = 1
            e[0, 2, 1] = e[2, 1, 0] = e[1, 0, 2] = -1
            omga = v = np.einsum('ijk,jk...->i...', e, S)

            vm = mu*np.sum(np.square(omga), axis=0)
            scalar_analysis(analyzer, vm, (emin[iv], emax[iv]), emean[iv],
                            None, None, 'mu_enst', 'Enstrophic dissipation',
                            '\mu\omega_i\omega_i')
            iv+=1

            S = S + np.rollaxis(S, 1)  # 2Sij
            Smm = vm = 0.5*np.einsum('mm...', S)

            v[0] = analyzer.div(rho*u[0]*u)
            v[1] = analyzer.div(rho*u[1]*u)
            v[2] = analyzer.div(rho*u[2]*u)
            A1 = np.sum(u*v, axis=0)
            scalar_analysis(analyzer, A1, (emin[iv], emax[iv]), emean[iv],
                            None, None, 'A1', 'A1', A1str)
            iv+=1

            v[0] = analyzer.div(u[0]*u)
            v[1] = analyzer.div(u[1]*u)
            v[2] = analyzer.div(u[2]*u)
            A2 = np.sum(rho*u*v, axis=0)
            scalar_analysis(analyzer, A2, (emin[iv], emax[iv]), emean[iv],
                            None, None, 'A2', 'A2', A2str)
            iv+=1

            A3 = rho*np.sum(np.square(u), axis=0)*Smm
            scalar_analysis(analyzer, A3, (emin[iv], emax[iv]), emean[iv],
                            None, None, 'A3', 'A3', A3str)
            iv+=1

            divKu = analyzer.div(rho*np.sum(np.square(u), axis=0)*u)
            RHS = -divKu
            scalar_analysis(analyzer, divKu, (emin[iv], emax[iv]), emean[iv],
                            None, None, 'divKu', 'divKu', Astr)
            iv+=1

            res = A3 = divKu - A1 - A2 + A3
            scalar_analysis(analyzer, res, (emin[iv], emax[iv]), emean[iv],
                            None, None, 'Ares', 'residual advective flux',
                            Ares)
            iv+=1

            P1 = A1 = np.sum(u*analyzer.scl_grad(P), axis=0)
            scalar_analysis(analyzer, P1, (emin[iv], emax[iv]), emean[iv],
                            None, None, 'P1', 'P1', P1str)
            iv+=1

            Pdil = A2 = P*Smm
            RHS += Pdil
            scalar_analysis(analyzer, Pdil, (emin[iv], emax[iv]), emean[iv],
                            None, None, 'Pdil', 'Pdil', Pdilstr)
            iv+=1

            divPu = divKu = analyzer.div(P*u)
            RHS -= divPu
            scalar_analysis(analyzer, divPu, (emin[iv], emax[iv]), emean[iv],
                            None, None, 'divPu', 'divPu', Pstr)
            iv+=1

            res = divPu - P1 - Pdil
            scalar_analysis(analyzer, res, (emin[iv], emax[iv]), emean[iv],
                            None, None, 'Pres', 'residual pressure flux', Pres)
            iv+=1

            v[0] = analyzer.div(mu*S[0])
            v[1] = analyzer.div(mu*S[1])
            v[2] = analyzer.div(mu*S[2])
            D1 = A1 = np.sum(u*v, axis=0)
            scalar_analysis(analyzer, D1, (emin[iv], emax[iv]), emean[iv],
                            None, None, 'D1', 'D1', D1str)
            iv+=1

            D2 = A2 = 0.5*mu*np.sum(np.square(S), axis=(0, 1))
            D2 += mu*np.square(Smm)
            RHS -= D2
            scalar_analysis(analyzer, D2, (emin[iv], emax[iv]), emean[iv],
                            None, None, 'D2', 'D2', D2str)
            iv+=1

            divSu = divKu = analyzer.div(mu*(np.sum(S*u, axis=0)+u*Smm))
            RHS += divSu
            scalar_analysis(analyzer, divSu, (emin[iv], emax[iv]), emean[iv],
                            None, None, 'divSu', 'divSu', Dstr)
            iv+=1

            res = divSu - D1 - D2
            scalar_analysis(analyzer, res, (emin[iv], emax[iv]), emean[iv],
                            None, None, 'Dres', 'residual diffusive flux',
                            Dres)
            iv+=1

            # RHS
            scalar_analysis(analyzer, RHS, (emin[iv], emax[iv]), emean[iv],
                            None, None, 'RHS', 'RHS of kinetic energy budget',
                            '\mathrm{{RHS}}')

            comm.Barrier()
            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'\t\tanalysis loop finished')
            # -----------------------------------------------------------------

        if comm.rank % 64 == 0:
            print (update(timeofday(), tstep, 0, comm.rank)
                   +'\ttime loop finished')
        # ---------------------------------------------------------------------

    # -------------------------------------------------------------------------

    # tmean /= nt

    # for it in xrange(its, ite+1, tint):
    #     tstep = str(it).zfill(4)

    if comm.rank == 0:
        print("Python MPI job `physical_transport_analysis' finished at {}"
              .format(timeofday()))

    return


###############################################################################
if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: .8e}'.format})
    physical_transport_analysis(get_inputs())
