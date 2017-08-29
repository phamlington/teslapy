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
# import hashlib
comm = MPI.COMM_WORLD


###############################################################################
def dns_statistical_analysis(args):
    if comm.rank == 0:
        print("Python MPI job `dns_statistical_analysis' started with "
              "{} tasks at {}.".format(comm.size, timeofday()))

    (idir, odir, pid, N, L, irs, ire, rint, its, ite, tint, gamma, R,
     texp, tcoef, T_0) = args

    nr = (ire-irs)/rint + 1
    # nt = (ite-its)/tint + 1
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

    s = list(analyzer.nnx)
    s.insert(0, 3)
    analyzer.tol = 1.0e-16

    psum = mpiAnalyzer.psum
    MMIN = MPI.MIN
    MMAX = MPI.MAX
    MSUM = MPI.SUM

    e = np.zeros((3, 3, 3))
    e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = 1
    e[0, 2, 1] = e[2, 1, 0] = e[1, 0, 2] = -1

    dTstr = '\\frac{\partial T}{\partial x_i}'
    dTdTstr = '{0}{0}'.format(dTstr)
    sclstr='\\rho\\varepsilon_T'
    sstr="\\rho\\varepsilon^\mathrm{s}"
    dstr="\\rho\\varepsilon^\mathrm{d}"
    Ek_fmt = "\widehat{{{0}}}^*\widehat{{{0}}}".format

    # -------------------------------------------------------------------------
    # Start first set of loops to determine ensemble and temporal mins, maxs,
    # and means

    # tmin = np.zeros(31)
    # tmax = np.zeros(31)
    # tmean= np.zeros(53)

    for it in xrange(its, ite+1, tint):
        tstep = str(it).zfill(4)

        emin = np.empty(50)
        emax = np.empty(50)

        emin[:] = np.inf
        emax[:] = np.NINF

        for ir in xrange(irs, ire+1, rint):

            u = np.empty(s, dtype=np.float64)
            rho = reader.get_variable(fmt(prefix[0], tstep, ir))
            u[0] = reader.get_variable(fmt(prefix[1], tstep, ir))
            u[1] = reader.get_variable(fmt(prefix[2], tstep, ir))
            u[2] = reader.get_variable(fmt(prefix[3], tstep, ir))
            re = reader.get_variable(fmt(prefix[4], tstep, ir))

            iv = 0

            emin[iv] = min(emin[iv], comm.allreduce(np.min(rho), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(rho), op=MMAX))
            iv+=1

            # emin[iv] = min(emin[iv], comm.allreduce(np.min(u), op=MMIN))
            # emax[iv] = max(emax[iv], comm.allreduce(np.max(u), op=MMAX))

            vm = np.sum(np.square(u), axis=0)
            emin[iv] = min(emin[iv], comm.allreduce(np.min(vm), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(vm), op=MMAX))
            iv+=1

            vm *= 0.5*rho
            rekin = vm
            emin[iv] = min(emin[iv], comm.allreduce(np.min(rekin), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(rekin), op=MMAX))
            iv+=1

            # v = rho*u
            # emin[iv] = min(emin[iv], comm.allreduce(np.min(v), op=MMIN))
            # emax[iv] = max(emax[iv], comm.allreduce(np.max(v), op=MMAX))
            # iv+=1

            # v = np.sqrt(rho)*u
            # emin[iv] = min(emin[iv], comm.allreduce(np.min(v), op=MMIN))
            # emax[iv] = max(emax[iv], comm.allreduce(np.max(v), op=MMAX))
            # iv+=1

            re -= rekin
            re *= gamma-1
            P = re
            emin[iv] = min(emin[iv], comm.allreduce(np.min(P), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(P), op=MMAX))
            iv+=1

            T = P/(rho*R)
            emin[iv] = min(emin[iv], comm.allreduce(np.min(T), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(T), op=MMAX))
            iv+=1

            c_s = np.sqrt(gamma*R*T)
            # emin[iv] = min(emin[iv], comm.allreduce(np.min(c_s), op=MMIN))
            # emax[iv] = max(emax[iv], comm.allreduce(np.max(c_s), op=MMAX))
            # iv+=1

            vm = np.sqrt(np.sum(np.square(u), axis=0))/c_s
            M = vm
            emin[iv] = min(emin[iv], comm.allreduce(np.min(M), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(M), op=MMAX))
            iv+=1

            mu = tcoef*np.power(T, texp)
            # emin[iv] = min(emin[iv], comm.allreduce(np.min(mu), op=MMIN))
            # emax[iv] = max(emax[iv], comm.allreduce(np.max(mu), op=MMAX))
            # iv+=1

            v = analyzer.scl_grad(T)
            gradT = v
            # emin[iv] = min(emin[iv], comm.allreduce(np.min(gradT), op=MMIN))
            # emax[iv] = max(emax[iv], comm.allreduce(np.max(gradT), op=MMAX))
            # iv+=1

            vm = np.sum(np.square(gradT), axis=0)
            dTdT = vm
            emin[iv] = min(emin[iv], comm.allreduce(np.min(dTdT), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(dTdT), op=MMAX))
            iv+=1

            dTdT *= mu
            repsT = dTdT
            emin[iv] = min(emin[iv], comm.allreduce(np.min(repsT), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(repsT), op=MMAX))
            iv+=1

            A = analyzer.grad(u)

            v = np.einsum('ijk,jk...->i...', e, A)
            omga = v
            # emin[iv] = min(emin[iv], comm.allreduce(np.min(omga), op=MMIN))
            # emax[iv] = max(emax[iv], comm.allreduce(np.max(omga), op=MMAX))
            # iv+=1

            vm = 0.5*np.sum(np.square(omga), axis=0)
            enst = vm
            emin[iv] = min(emin[iv], comm.allreduce(np.min(enst), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(enst), op=MMAX))
            iv+=1

            A = 0.5*(A + np.rollaxis(A, 1))
            S = A
            emin[iv] = min(emin[iv], comm.allreduce(np.min(S), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(S), op=MMAX))
            iv+=1

            Smm = np.einsum('ii...', S)
            emin[iv] = min(emin[iv], comm.allreduce(np.min(Smm), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(Smm), op=MMAX))
            iv+=1

            dil2 = np.square(Smm)
            emin[iv] = min(emin[iv], comm.allreduce(np.min(dil2), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(dil2), op=MMAX))
            iv+=1

            S2 = np.sum(np.square(S), axis=(0, 1)) - (1./3.)*dil2
            emin[iv] = min(emin[iv], comm.allreduce(np.min(S2), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(S2), op=MMAX))
            iv+=1

            # divKu = rekin = analyzer.div(2.0*rekin*u)
            # emin[iv] = min(emin[iv], comm.allreduce(np.min(divKu), op=MMIN))
            # emax[iv] = max(emax[iv], comm.allreduce(np.max(divKu), op=MMAX))
            # iv+=1
            # RHS = divKu
            # RHS *= -1

            # divPu = vm = analyzer.div(P*u)
            # RHS -= divPu
            # emin[iv] = min(emin[iv], comm.allreduce(np.min(divPu), op=MMIN))
            # emax[iv] = max(emax[iv], comm.allreduce(np.max(divPu), op=MMAX))
            # iv+=1

            S2 *= 2.0*mu
            repss = S2
            emin[iv] = min(emin[iv], comm.allreduce(np.min(repss), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(repss), op=MMAX))
            iv+=1

            dil2 *= mu
            repsd = dil2
            emin[iv] = min(emin[iv], comm.allreduce(np.min(repsd), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(repsd), op=MMAX))
            iv+=1

            # reps = vm = repsd + repss
            # RHS -= reps
            # emin[iv] = min(emin[iv], comm.allreduce(np.min(reps), op=MMIN))
            # emax[iv] = max(emax[iv], comm.allreduce(np.max(reps), op=MMAX))
            # iv+=1

            vm = P*Smm
            Pdil = vm
            # RHS += Pdil
            emin[iv] = min(emin[iv], comm.allreduce(np.min(Pdil), op=MMIN))
            emax[iv] = max(emax[iv], comm.allreduce(np.max(Pdil), op=MMAX))
            iv+=1

            # # RHS
            # emin[iv] = min(emin[iv], comm.allreduce(np.min(RHS), op=MMIN))
            # emax[iv] = max(emax[iv], comm.allreduce(np.max(RHS), op=MMAX))
            # iv+=1

            comm.Barrier()
            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'min/max loop finished')

            # -----------------------------------------------------------------

        rhom = comm.allreduce(psum(rho), op=MSUM)/Ne

        # for i in xrange(29):
        #     tmin[i] = min(tmin[i], emin[i])
        #     tmax[i] = max(tmax[i], emax[i])

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

            u = np.empty(s, dtype=np.float64)
            rho = reader.get_variable(fmt(prefix[0], tstep, ir))
            u[0] = reader.get_variable(fmt(prefix[1], tstep, ir))
            u[1] = reader.get_variable(fmt(prefix[2], tstep, ir))
            u[2] = reader.get_variable(fmt(prefix[3], tstep, ir))
            re = reader.get_variable(fmt(prefix[4], tstep, ir))

            iv = 0

            scalar_analysis(analyzer, rho, (emin[iv], emax[iv]), None, None,
                            'rho', 'density', '\\rho')
            iv+=1

            analyzer.spectral_density(u, 'u', 'velocity PSD', Ek_fmt('u_i'))

            vm = np.sum(np.square(u), axis=0)
            scalar_analysis(analyzer, vm, (emin[iv], emax[iv]), None, None,
                            'uiui', 'velocity squared', 'u_iu_i')
            scalar_analysis(analyzer, vm, (emin[iv], emax[iv]),
                            rho.copy(), rhom,
                            'uiui_tilde', 'm.w. velocity squared', 'u_iu_i')
            iv+=1

            vm *= 0.5*rho
            rekin = vm
            scalar_analysis(analyzer, rekin, (emin[iv], emax[iv]), None, None,
                            'rekin', 'kinetic energy', '\\rho u_iu_i/2')
            iv+=1

            v = rho*u
            analyzer.spectral_density(v, 'ru', 'momentum PSD',
                                      Ek_fmt('\\rho u_i'))

            v = np.sqrt(rho)*u
            analyzer.spectral_density(v, 'v', 'spectral kinetic energy',
                                      Ek_fmt('\sqrt{\\rho} u_i'))

            comm.Barrier()

            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'\tvelocity analyses completed')
            # -----------------------------------------------------------------

            re -= rekin
            re *= gamma-1
            P = re

            if P.min() < 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'\tP check: {} {}'.format(P.min(), P.max()))

            scalar_analysis(analyzer, P, (emin[iv], emax[iv]), None, None,
                            'P', 'pressure', 'P')
            iP = iv
            iv+=1

            T = P/(rho*R)
            scalar_analysis(analyzer, T, (emin[iv], emax[iv]), None, None,
                            'T', 'temperature', 'T')
            scalar_analysis(analyzer, T, (emin[iv], emax[iv]), rho, rhom,
                            'T_tilde', 'm.w. temperature', 'T')
            iT = iv
            iv+=1

            # range1 = (emin[0], emax[0])
            # range2 = (emin[iP], emax[iP])
            # analyzer.mpi_histogram2(rho, P, 'rho_P', '\\rho', 'P',
            #                         range1, range2)

            # range2 = (emin[iT], emax[iT])
            # analyzer.mpi_histogram2(rho, T, 'rho_T', '\\rho', 'T',
            #                         range1, range2)

            # range1 = (emin[iP], emax[iP])
            # analyzer.mpi_histogram2(P, T, 'P_T_tilde', 'P', 'T',
            #                         range1, range2, 100, rho)
            # analyzer.mpi_histogram2(P, T, 'P_T', 'P', 'T',
            #                         range1, range2)

            c_s = np.sqrt(gamma*R*T)
            analyzer.write_mpi_moments(c_s, 'speed of sound', 'c_\\mathrm{s}',
                                       None, None, m1=0)
            analyzer.write_mpi_moments(c_s, 'm.w. speed of sound',
                                       'c_\\mathrm{s}', rho, rhom, m1=0)

            vm = np.sqrt(np.sum(np.square(u), axis=0))/c_s
            M = vm
            scalar_analysis(analyzer, M, (emin[iv], emax[iv]), None, None,
                            'M', 'Mach number', '\mathrm{Ma}')
            scalar_analysis(analyzer, M, (emin[iv], emax[iv]), rho, rhom,
                            'M_tilde', 'm.w. Mach number', '\mathrm{Ma}')
            iv+=1

            mu = tcoef*np.power(T, texp)
            analyzer.write_mpi_moments(mu, 'viscosity', '\mu',
                                       None, None, m1=0)
            analyzer.write_mpi_moments(mu, 'm.w. viscosity', '\mu',
                                       rho, rhom, m1=0)

            # comm.Barrier()
            # if comm.rank % 64 == 0:
            #     print (update(timeofday(), tstep, ir, comm.rank)
            #            +'\tthermodynamic analyses completed')
            # -----------------------------------------------------------------

            v = analyzer.scl_grad(T)
            gradT = v
            analyzer.spectral_density(
                    gradT, 'gradT', 'temperature gradient PSD', Ek_fmt(dTstr))

            vm = np.sum(np.square(gradT), axis=0)
            dTdT = vm
            scalar_analysis(analyzer, dTdT, (emin[iv], emax[iv]), None, None,
                            'dTdT', 'temperature gradient squared', dTdTstr)
            scalar_analysis(analyzer, dTdT, (emin[iv], emax[iv]), rho, rhom,
                            'dTdT_tilde', 'm.w. temperature gradient squared',
                            dTdTstr)
            idT=iv
            iv+=1

            dTdT *= mu
            repsT = dTdT
            scalar_analysis(analyzer, repsT, (emin[iv], emax[iv]), None, None,
                            'repsT', 'scalar dissipation', sclstr)
            iv+=1

            # range1 = (emin[iT], emax[iT])
            # range2 = (emin[idT+1], emax[idT+1])
            # analyzer.mpi_histogram2(T, repsT, 'T_repsT_tilde', 'T',
            #                         sclstr, range1, range2, 100, rho)
            # analyzer.mpi_histogram2(T, repsT, 'T_repsT', 'T', sclstr,
            #                         range1, range2)

            comm.Barrier()
            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'\ttemperature gradient analyses completed')
            # -----------------------------------------------------------------

            A = analyzer.grad(u)

            comm.Barrier()
            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'\tvelocity gradients computed')
            # -----------------------------------------------------------------

            v = np.einsum('ijk,jk...->i...', e, A)
            omga = v
            analyzer.spectral_density(omga, 'omga', 'vorticity PSD',
                                      Ek_fmt('\omega_i'))

            vm = 0.5*np.sum(np.square(omga), axis=0)
            enst = vm
            scalar_analysis(analyzer, enst, (emin[iv], emax[iv]), None, None,
                            'enst', 'enstrophy', '\Omega')
            scalar_analysis(analyzer, enst, (emin[iv], emax[iv]), rho, rhom,
                            'enst_tilde', 'm.w. enstrophy', '\Omega')
            iOm=iv
            iv+=1

            # range2 = (emin[iOm], emax[iOm])
            # analyzer.mpi_histogram2(T, enst, 'T_enst_tilde', 'T', '\Omega',
            #                         range1, range2, 100, rho)
            # analyzer.mpi_histogram2(T, enst, 'T_enst', 'T', '\Omega',
            #                         range1, range2)

            # range1 = (emin[iP], emax[iP])
            # analyzer.mpi_histogram2(P, enst, 'P_enst_tilde', 'P', '\Omega',
            #                         range1, range2, 100, rho)
            # analyzer.mpi_histogram2(P, enst, 'P_enst', 'P', '\Omega',
            #                         range1, range2)

            # range1 = (emin[0], emax[0])
            # analyzer.mpi_histogram2(rho, enst, 'rho_enst', '\\rho', '\Omega',
            #                         range1, range2)

            # range1 = (emin[iOm], emax[iOm])
            # range2 = (emin[idT], emax[idT])
            # analyzer.mpi_histogram2(enst, dTdT, 'enst_dTdT', dTdTstr,
            #                         '\Omega', range1, range2)
            # analyzer.mpi_histogram2(enst, dTdT, 'enst_dTdT_tilde', dTdTstr,
            #                         '\Omega', range1, range2, 100, rho)

            A = 0.5*(A + np.rollaxis(A, 1))
            S = A
            gradient_analysis(analyzer, S, (emin[iv], emax[iv]), None, None,
                              'S', 'strain rate', 'S')
            gradient_analysis(analyzer, S, (emin[iv], emax[iv]), rho, rhom,
                              'S_tilde', 'm.w. strain rate', 'S')
            iv+=1

            comm.Barrier()
            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'\tvelocity gradient analyses completed')
            # -----------------------------------------------------------------

            Smm = np.einsum('ii...', S)
            scalar_analysis(analyzer, Smm, (emin[iv], emax[iv]), None, None,
                            'Smm', 'dilatation', '\Theta')
            scalar_analysis(analyzer, Smm, (emin[iv], emax[iv]), rho, rhom,
                            'Smm_tilde', 'm.w. dilatation', '\Theta')
            iD=iv
            iv+=1

            # range1 = (emin[iT], emax[iT])
            # range2 = (emin[iD], emax[iD])
            # analyzer.mpi_histogram2(T, Smm, 'T_Smm_tilde', 'T', '\Theta',
            #                         range1, range2, 100, rho)
            # analyzer.mpi_histogram2(T, Smm, 'T_Smm', 'T', '\Theta',
            #                         range1, range2)

            # range1 = (emin[iP], emax[iP])
            # analyzer.mpi_histogram2(P, Smm, 'P_Smm_tilde', 'P', '\Theta',
            #                         range1, range2, 100, rho)
            # analyzer.mpi_histogram2(P, Smm, 'P_Smm', 'P', '\Theta',
            #                         range1, range2)

            # range1 = (emin[0], emax[0])
            # analyzer.mpi_histogram2(rho, Smm, 'rho_Smm', '\\rho', '\Theta',
            #                         range1, range2)

            dil2 = np.square(Smm)
            scalar_analysis(analyzer, dil2, (emin[iv], emax[iv]), None, None,
                            'dil2', 'dilatation squared', '\Theta^2')
            scalar_analysis(analyzer, dil2, (emin[iv], emax[iv]), rho, rhom,
                            'dil2_tilde', 'm.w. dilatation squared',
                            '\Theta^2')
            iD2=iv
            iv+=1

            # range1 = (emin[iOm], emax[iOm])
            # range2 = (emin[iD2], emax[iD2])
            # analyzer.mpi_histogram2(enst, dil2, 'enst_dil2_tilde',
            #                         '\Omega', '\Theta^2',
            #                         range1, range2, 100, rho)
            # analyzer.mpi_histogram2(enst, dil2, 'enst_dil2',
            #                         '\Omega', '\Theta^2', range1, range2)

            # range1 = (emin[iT], emax[iT])
            # analyzer.mpi_histogram2(T, dil2, 'T_dil2_tilde', 'T', '\Theta^2',
            #                         range1, range2, 100, rho)
            # analyzer.mpi_histogram2(T, dil2, 'T_dil2', 'T', '\Theta^2',
            #                         range1, range2)

            # range1 = (emin[iP], emax[iP])
            # analyzer.mpi_histogram2(P, dil2, 'P_dil2_tilde', 'P', '\Theta^2',
            #                         range1, range2, 100, rho)
            # analyzer.mpi_histogram2(P, dil2, 'P_dil2', 'P', '\Theta^2',
            #                         range1, range2)

            # range1 = (emin[0], emax[0])
            # analyzer.mpi_histogram2(rho, dil2, 'rho_dil2', '\\rho',
            #                         '\Theta^2', range1, range2)

            # range1 = (emin[iD2], emax[iD2])
            # range2 = (emin[idT], emax[idT])
            # analyzer.mpi_histogram2(dil2, dTdT, 'dil2_dTdT_tilde',
            #                         '\Theta^2', dTdTstr,
            #                         range1, range2, 100, rho)
            # analyzer.mpi_histogram2(dil2, dTdT, 'dil2_dTdT',
            #                         '\Theta^2', dTdTstr, range1, range2)

            S2 = np.sum(np.square(S), axis=(0, 1)) - (1./3.)*dil2
            scalar_analysis(analyzer, S2, (emin[iv], emax[iv]), None, None,
                            'S2', 'traceless strain squared', "S_{tl}^2")
            scalar_analysis(analyzer, S2, (emin[iv], emax[iv]), rho, rhom,
                            'S2_tilde', 'm.w. traceless strain squared',
                            "S_{tl}^2")
            iS2=iv
            iv+=1

            # range1 = (emin[iOm], emax[iOm])
            # range2 = (emin[iS2], emax[iS2])
            # analyzer.mpi_histogram2(enst, S2, 'enst_S2_tilde',
            #                         '\Omega', "S_{tl}^2",
            #                         range1, range2, 100, rho)
            # analyzer.mpi_histogram2(enst, S2, 'enst_S2', '\Omega',
            #                         "S_{tl}^2", range1, range2)

            # range1 = (emin[iD2], emax[iD2])
            # analyzer.mpi_histogram2(dil2, S2, 'dil2_S2_tilde',
            #                         '\Omega', "S_{tl}^2",
            #                         range1, range2, 100, rho)
            # analyzer.mpi_histogram2(dil2, S2, 'dil2_S2', '\Omega',
            #                         "S_{tl}^2", range1, range2)

            # range1 = (emin[iT], emax[iT])
            # analyzer.mpi_histogram2(T, S2, 'T_S2_tilde', 'T', 'S_{tl}^2',
            #                         range1, range2, 100, rho)
            # analyzer.mpi_histogram2(T, S2, 'T_S2', 'T', 'S_{tl}^2',
            #                         range1, range2)

            # range1 = (emin[iP], emax[iP])
            # analyzer.mpi_histogram2(P, S2, 'P_S2_tilde', 'P', 'S_{tl}^2',
            #                         range1, range2, 100, rho)
            # analyzer.mpi_histogram2(P, S2, 'P_S2', 'P', 'S_{tl}^2',
            #                         range1, range2)

            # range1 = (emin[0], emax[0])
            # analyzer.mpi_histogram2(rho, S2, 'rho_S2', '\\rho', 'S_{tl}^2',
            #                         range1, range2)

            comm.Barrier()
            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'\tstrain analyses completed')
            # -----------------------------------------------------------------

            S2 *= 2.0*mu
            repss = S2
            scalar_analysis(analyzer, repss, (emin[iv], emax[iv]), None, None,
                            'repss', 'sol. KE dissipation', sstr)
            iS=iv
            iv+=1

            # range1 = (emin[iT], emax[iT])
            # range2 = (emin[iS], emax[iS])
            # analyzer.mpi_histogram2(T, repss, 'T_repss_tilde', 'T', sstr,
            #                         range1, range2, 100, rho)
            # analyzer.mpi_histogram2(T, repss, 'T_repss', 'T', sstr,
            #                         range1, range2)

            # range1 = (emin[iP], emax[iP])
            # analyzer.mpi_histogram2(P, repss, 'P_repss', 'P', sstr,
            #                         range1, range2)

            # range1 = (emin[0], emax[0])
            # analyzer.mpi_histogram2(rho, repss, 'rho_repss', '\\rho', sstr,
            #                         range1, range2)

            dil2 *= mu
            repsd = dil2
            scalar_analysis(analyzer, repsd, (emin[iv], emax[iv]), None, None,
                            'repsd', 'dil. KE dissipation', dstr)
            iD=iv
            iv+=1

            # range1 = (emin[iT], emax[iT])
            # range2 = (emin[iD], emax[iD])
            # analyzer.mpi_histogram2(T, repsd, 'T_repsd_tilde', 'T', dstr,
            #                         range1, range2, 100, rho)
            # analyzer.mpi_histogram2(T, repsd, 'T_repsd', 'T', dstr,
            #                         range1, range2)

            # range1 = (emin[iP], emax[iP])
            # analyzer.mpi_histogram2(P, repsd, 'P_repsd', 'P', dstr,
            #                         range1, range2)

            # range1 = (emin[0], emax[0])
            # analyzer.mpi_histogram2(rho, repsd, 'rho_repsd', '\\rho', dstr,
            #                         range1, range2)

            # reps = mu = repsd + repss
            # scalar_analysis(analyzer, reps, (emin[iv], emax[iv]), None, None,
            #                 'reps', "KE dissipation", "\\rho\\varepsilon")
            # iR=iv
            # iv+=1

            # range1 = (emin[iT], emax[iT])
            # range2 = (emin[iR], emax[iR])
            # analyzer.mpi_histogram2(T, reps, 'T_reps_tilde', 'T',
            #                         "\\rho\\varepsilon", range1, range2,
            #                         100, rho)
            # analyzer.mpi_histogram2(T, reps, 'T_reps', 'T',
            #                         "\\rho\\varepsilon", range1, range2)

            # range1 = (emin[iP], emax[iP])
            # analyzer.mpi_histogram2(P, reps, 'P_reps', 'P',
            #                         "\\rho\\varepsilon", range1, range2)

            # range1 = (emin[0], emax[0])
            # analyzer.mpi_histogram2(rho, reps, 'rho_reps', '\\rho',
            #                         "\\rho\\varepsilon", range1, range2)

            vm = P*Smm
            Pdil = vm
            scalar_analysis(analyzer, Pdil, (emin[iv], emax[iv]), None, None,
                            'Pdil', 'pressure-dilatation', 'P\Theta')
            iPd=iv
            iv+=1

            # range1 = (emin[iT], emax[iT])
            # range2 = (emin[iPd], emax[iPd])
            # analyzer.mpi_histogram2(T, Pdil, 'T_Pdil_tilde', 'T', 'P\Theta',
            #                         range1, range2,
            #                         100, rho)
            # analyzer.mpi_histogram2(T, Pdil, 'T_Pdil', 'T', 'P\Theta',
            #                         range1, range2)

            # range1 = (emin[iP], emax[iP])
            # analyzer.mpi_histogram2(P, Pdil, 'P_Pdil', 'P', 'P\Theta',
            #                         range1, range2)

            # range1 = (emin[0], emax[0])
            # analyzer.mpi_histogram2(rho, Pdil, 'rho_Pdil', '\\rho',
            #                         'P\Theta', range1, range2)

            # RHS = vm = -divKu - divPu + divSu + Pdil - reps
            # scalar_analysis(analyzer, RHS, (emin[iv], emax[iv]), None, None,
            #                 'RHS', "total KE RHS",
            #                 "\epsilon")

            # range1 = (emin[iT], emax[iT])
            # range2 = (emin[iv], emax[iv])
            # analyzer.mpi_histogram2(T, RHS, 'T_RHS_tilde', 'T', '\epsilon',
            #                         range1, range2,
            #                         100, rho)
            # analyzer.mpi_histogram2(T, RHS, 'T_RHS', 'T', '\epsilon',
            #                         range1, range2)

            # range1 = (emin[iP], emax[iP])
            # analyzer.mpi_histogram2(P, RHS, 'P_RHS', 'P', '\epsilon',
            #                         range1, range2)

            # range1 = (emin[0], emax[0])
            # analyzer.mpi_histogram2(rho, RHS, 'rho_RHS', '\\rho', '\epsilon',
            #                         range1, range2)

            # comm.Barrier()
            # if comm.rank % 64 == 0:
            #     print (update(timeofday(), tstep, ir, comm.rank)
            #            +'\tdissipation analyses completed')
            # -----------------------------------------------------------------

            # range1 = (emin[idT+1], emax[idT+1])
            # range2 = (emin[iS], emax[iS])
            # analyzer.mpi_histogram2(repsT, repss, 'repsT_repss',
            #                         sclstr, sstr, range1, range2)

            # range2 = (emin[iD], emax[iD])
            # analyzer.mpi_histogram2(repsT, repsd, 'repsT_repsd',
            #                         sclstr, dstr, range1, range2)

            # range2 = (emin[iv], emax[iv])
            # analyzer.mpi_histogram2(repsT, reps, 'repsT_reps', sclstr,
            #                         '\\rho\\varepsilon', range1, range2)

            # range2 = (emin[iPd], emax[iPd])
            # analyzer.mpi_histogram2(repsT, Pdil, 'repsT_Pdil',
            #                         sclstr, 'P\Theta', range1, range2)

            # range2 = (emin[iv], emax[iv])
            # analyzer.mpi_histogram2(repsT, RHS, 'repsT_RHS',
            #                         sclstr, '\epsilon', range1, range2)

            # range1 = (emin[iS], emax[iS])
            # range2 = (emin[iD], emax[iD])
            # analyzer.mpi_histogram2(repss, repsd, 'repss_repsd', sstr,
            #                         dstr, range1, range2)

            # range2 = (emin[iR], emax[iR])
            # analyzer.mpi_histogram2(repss, reps, 'repss_reps',
            #                         sstr, '\\rho\\varepsilon',
            #                         range1, range2)

            # range2 = (emin[iPd], emax[iPd])
            # analyzer.mpi_histogram2(repss, Pdil, 'repss_Pdil',
            #                         sstr, 'P\Theta', range1, range2)

            # range2 = (emin[iv], emax[iv])
            # analyzer.mpi_histogram2(repss, RHS, 'repss_RHS',
            #                         sstr, '\epsilon', range1, range2)

            # range2 = (emin[iR], emax[iR])
            # analyzer.mpi_histogram2(repsd, reps, 'repsd_reps',
            #                         dstr, '\\rho\\varepsilon',
            #                         range1, range2)

            # range2 = (emin[iPd], emax[iPd])
            # analyzer.mpi_histogram2(repsd, Pdil, 'repsd_Pdil',
            #                         dstr, 'P\Theta', range1, range2)

            # range2 = (emin[iv], emax[iv])
            # analyzer.mpi_histogram2(repsd, RHS, 'repsd_RHS',
            #                         dstr, '\epsilon', range1, range2)

            # range1 = (emin[iR], emax[iR])
            # range2 = (emin[iPd], emax[iPd])
            # analyzer.mpi_histogram2(reps, Pdil, 'reps_Pdil',
            #                         '\\rho\\varepsilon', 'P\Theta',
            #                         range1, range2)

            # range2 = (emin[iv], emax[iv])
            # analyzer.mpi_histogram2(reps, RHS, 'reps_RHS',
            #                         '\\rho\\varepsilon', '\epsilon',
            #                         range1, range2)

            # range1 = (emin[iPd], emax[iPd])
            # range2 = (emin[iv], emax[iv])
            # analyzer.mpi_histogram2(Pdil, RHS, 'Pdil_RHS',
            #                         'P\Theta', '\epsilon',
            #                         range1, range2)

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
        print("Python MPI job `dns_statistical_analysis' finished at {}"
              .format(timeofday()))

    return


###############################################################################
if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: .8e}'.format})
    dns_statistical_analysis(get_inputs())
