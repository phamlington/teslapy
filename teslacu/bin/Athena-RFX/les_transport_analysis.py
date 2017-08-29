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
import gc
# from memory_profiler import profile
# import os
# import time
comm = MPI.COMM_WORLD


###############################################################################
# @profile
def filtered_transport_analysis(args):
    if comm.rank == 0:
        print("Python MPI job `filtered_transport_analysis' started with "
              "{} tasks at {}.".format(comm.size, timeofday()))

    (idir, odir, pid, N, L, irs, ire, rint, its, ite, tint, gamma, R,
     texp, tcoef, tmp0) = args

    nr = (ire-irs)/rint + 1
    nt = (ite-its)/tint + 1
    Ne = nr*N**3
    gamma1 = gamma - 1
    igamma1 = 1.0/gamma1

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
                idir=idir, odir=odir, probID=pid, L=L, nx=N, geo='hit',
                method='akima')

    analyzer.tol = 1.0e-16

    update = '{}\t{}\t{}\t{}\t{}\t{}'.format
    fname = '{}_{}_{}.bin'.format
    ddx = '\\frac{{\partial}}{{\partial x_j}}'

    MMIN = MPI.MIN
    MMAX = MPI.MAX
    MSUM = MPI.SUM
    psum = analyzer.psum
    MREDUCE = comm.allreduce
    scalar_filter = analyzer.scalar_filter
    vector_filter = analyzer.vector_filter

    s = list(analyzer.nnx)
    s.insert(0, 3)
    u = np.empty(s, dtype=np.float64)
    s.insert(0, 1)

    filter_labels = ['G8', 'G16', 'G32', 'G64', 'G128']
    filter_widths = [L/8, L/16, L/32, L/64, L/128]

    for Glabel, ell in zip(filter_labels, filter_widths):
        if comm.rank % 128 == 0:
            print(update(timeofday(), Glabel, comm.rank,
                         'LES filter iteration started', '', ''))

        Ghat = analyzer.filter_kernel(ell)

        nv = 40
        tmin = np.empty(nv)
        tmax = np.empty(nv)
        tmean= np.zeros(nv)
        tmavg= np.zeros(nv)

        tmin[:] = np.inf
        tmax[:] = np.NINF

        for it in xrange(its, ite+1, tint):
            tstep = str(it).zfill(4)
            if comm.rank % 128 == 0:
                print(update(timeofday(), Glabel, tstep, comm.rank,
                             'time iteration started for min, max, mean', ''))

            emin = np.empty(nv)
            emax = np.empty(nv)
            emean= np.zeros(nv)
            emavg= np.zeros(nv)

            emin[:] = np.inf
            emax[:] = np.NINF

            for ir in xrange(irs, ire+1, rint):
                run = str(ir).zfill(2)

                if comm.rank % 128 == 0:
                    print(update(timeofday(), Glabel, tstep, run, comm.rank,
                                 'run iteration started'))

                rho = reader.get_variable(fname(prefix[0], tstep, ir))
                u[0] = reader.get_variable(fname(prefix[1], tstep, ir))
                u[1] = reader.get_variable(fname(prefix[2], tstep, ir))
                u[2] = reader.get_variable(fname(prefix[3], tstep, ir))
                re = reader.get_variable(fname(prefix[4], tstep, ir))

                # -------------------------------------------------------------

                # LES solution fields
                P = re = gamma1*(re - rho*0.5*np.sum(np.square(u), axis=0))

                rho_bar = scalar_filter(rho, Ghat)
                ru_bar = vector_filter(rho*u, Ghat)
                P_bar = scalar_filter(P, Ghat)

                u_tilde = ru_bar = ru_bar/rho_bar

                # Inaccessible filtered fields
                # rek_bar = scalar_filter(rek, Ghat)

                # Derived DNS and LES fields
                # rek = rho*0.5*np.sum(np.square(u), axis=0)
                mu = tcoef*np.power(P/rho/R, texp)

                Sij = analyzer.grad(u)
                Sij = 0.5*(Sij + np.rollaxis(Sij, 1))

                dil = np.einsum('mm...', Sij)
                dil2 = np.square(dil)
                S2 = np.sum(np.square(Sij), axis=(0, 1))

                div_psi_int = analyzer.div(vector_filter(igamma1*P*u, Ghat))
                div_psi_Q = analyzer.div(vector_filter(
                                mu*analyzer.scl_grad(P/rho/R), Ghat))

                Phi_P = scalar_filter(P*dil, Ghat)
                Phi_Ds = scalar_filter(2.0*mu*(S2 - (1./3.)*dil2), Ghat)
                Phi_Dd = scalar_filter(5./3.*mu*dil2, Ghat)

                # # Inconsistently-filtered fields
                # u_bar = vector_filter(u)
                # T_bar = scalar_filter(T, Ghat)
                # dil_bar = scalar_filter(dil, Ghat)
                # dil2_bar= scalar_filter(np.square(dil), Ghat)

                if comm.rank % 128 == 0:
                    print(update(timeofday(), Glabel, tstep, run, comm.rank,
                                 'DNS fields computed and filtered'))
                # -------------------------------------------------------------

                rek_res = 0.5*rho_bar*np.sum(np.square(u_tilde), axis=0)
                # rek_sgs = rek_bar - rek_res
                # rek_tke = rek - rek_bar
                mu_res = tcoef*np.power(P_bar/(rho_bar*R), texp)

                Sij_res = Sij = analyzer.grad(u_tilde)
                Sij_res = 0.5*(Sij_res + np.rollaxis(Sij_res, 1))

                dil_res = np.einsum('mm...', Sij_res)
                dil2_res = dil2 = np.square(dil_res)
                S2_res = S2 = np.sum(np.square(Sij_res), axis=(0, 1))

                Phi_P-= P_bar*dil_res
                Phi_Ds-= 2.0*mu_res*(S2_res - (1./3.)*dil2_res)
                Phi_Dd-= 5./3.*mu_res*dil2_res

                div_utau_As = np.zeros_like(rho)
                tauS_As = np.zeros_like(rho)
                div_utau_Ds = np.zeros_like(rho)
                tauS_Ds = np.zeros_like(rho)

                for j in range(3):
                    for i in range(3):
                        # tau^A_ij = \overline{rho u_i u_j}
                        #            - \overline{rho}\tilde{u_i}\tilde{u}_j
                        tau = scalar_filter(rho*u[i]*u[j], Ghat)
                        tau-= rho_bar*u_tilde[i]*u_tilde[j]

                        div_utau_As+= analyzer.deriv(tau*u_tilde[i], dim=j)
                        tauS_As += tau*Sij_res[j, i]

                        # tau^D_ij = \overline{sigma}_ij - \breve{sigma}_ij
                        tau = mu*(analyzer.deriv(u[i], dim=j)
                                  + analyzer.deriv(u[j], dim=i))
                        tau = scalar_filter(tau, Ghat)
                        tau-= 2.0*mu_res*Sij_res[j, i]

                        div_utau_Ds+= analyzer.deriv(tau*u_tilde[i], dim=j)
                        tauS_Ds += tau*Sij_res[j, i]

                # tau^Ad_ij = 1/3 tau^A_mm delta_ij
                tau = scalar_filter(rho*np.sum(np.square(u), axis=0), Ghat)
                tau-= 2.0*rek_res

                div_utau_Ad = (1./3.)*analyzer.div(tau*u_tilde)
                div_utau_As-= div_utau_Ad

                tauS_Ad = (1./3.)*tau*dil_res
                tauS_As-= tauS_Ad

                # tau^Dd_ij = 5/3(\overline{mu*dil}
                #             - \breve{mu}\breve{dil})delta_ij
                tau = scalar_filter(mu*dil, Ghat)
                tau-= mu_res*dil_res

                div_utau_Dd = analyzer.div(tau*u_tilde)
                div_utau_Ds-= (2./3.)*div_utau_Dd
                div_utau_Dd*= 5./3.

                tauS_Dd = tau*dil_res
                tauS_Ds-= (2./3.)*tauS_Dd
                tauS_Dd*= 5./3.

                div_Ku_res = analyzer.div(rek_res*u_tilde)
                div_reiu_res = analyzer.div(igamma1*P_bar*u_tilde)
                div_Su_res = 2.0*analyzer.div(mu_res*np.sum(
                                        Sij_res*u_tilde.reshape(s), axis=1))
                div_Du_res = analyzer.div(mu_res*dil_res*u_tilde)
                div_Su_res-= (2./3.)*div_Du_res
                div_Q_res = analyzer.div(mu_res*analyzer.scl_grad(
                                                        P_bar/(rho_bar*R)))

                div_psi_int-= div_reiu_res
                div_psi_Q-= div_Q_res

                # clearing out huge chunk of memory
                tau = dil = mu = P = re = None
                Sij_res = Sij = None

                n = gc.collect()
                if comm.rank % 128 == 0:
                    print(update(timeofday(), Glabel, tstep, run, comm.rank,
                                 'LES fields computed'))
                # -------------------------------------------------------------

                iv = 0

                emin[iv] = min(emin[iv], MREDUCE(np.min(rho_bar), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(rho_bar), op=MMAX))
                emean[iv]+= MREDUCE(psum(rho_bar), op=MSUM)
                iv+=1

                emin[iv] = min(emin[iv], MREDUCE(np.min(dil_res), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(dil_res), op=MMAX))
                emean[iv]+= MREDUCE(psum(dil_res), op=MSUM)
                emavg[iv]+= MREDUCE(psum(rho*dil_res), op=MSUM)
                iv+=1

                emin[iv] = min(emin[iv], MREDUCE(np.min(dil2_res), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(dil2_res), op=MMAX))
                emean[iv]+= MREDUCE(psum(dil2_res), op=MSUM)
                emavg[iv]+= MREDUCE(psum(rho*dil2_res), op=MSUM)
                iv+=1

                emin[iv] = min(emin[iv], MREDUCE(np.min(S2_res), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(S2_res), op=MMAX))
                emean[iv]+= MREDUCE(psum(S2_res), op=MSUM)
                emavg[iv]+= MREDUCE(psum(rho*S2_res), op=MSUM)
                iv+=1

                # S_res
                vm = np.sqrt(S2_res)
                emin[iv] = min(emin[iv], MREDUCE(np.min(vm), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(vm), op=MMAX))
                emean[iv]+= MREDUCE(psum(vm), op=MSUM)
                emavg[iv]+= MREDUCE(psum(rho*vm), op=MSUM)
                iv+=1

                # S2s_res
                vm = S2_res - (1./3.)*dil2_res
                emin[iv] = min(emin[iv], MREDUCE(np.min(vm), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(vm), op=MMAX))
                emean[iv]+= MREDUCE(psum(vm), op=MSUM)
                emavg[iv]+= MREDUCE(psum(rho*vm), op=MSUM)
                iv+=1

                # Ss_res
                vm = np.sqrt(vm)
                emin[iv] = min(emin[iv], MREDUCE(np.min(vm), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(vm), op=MMAX))
                emean[iv]+= MREDUCE(psum(vm), op=MSUM)
                emavg[iv]+= MREDUCE(psum(rho*vm), op=MSUM)
                iv+=1

                div_utau_As *= -1.0
                emin[iv] = min(emin[iv], MREDUCE(np.min(div_utau_As), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(div_utau_As), op=MMAX))
                emean[iv]+= MREDUCE(psum(div_utau_As), op=MSUM)
                iv+=1

                div_utau_Ad *= -1.0
                emin[iv] = min(emin[iv], MREDUCE(np.min(div_utau_Ad), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(div_utau_Ad), op=MMAX))
                emean[iv]+= MREDUCE(psum(div_utau_Ad), op=MSUM)
                iv+=1

                emin[iv] = min(emin[iv], MREDUCE(np.min(tauS_As), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(tauS_As), op=MMAX))
                emean[iv]+= MREDUCE(psum(tauS_As), op=MSUM)
                iv+=1

                emin[iv] = min(emin[iv], MREDUCE(np.min(tauS_Ad), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(tauS_Ad), op=MMAX))
                emean[iv]+= MREDUCE(psum(tauS_Ad), op=MSUM)
                iv+=1

                emin[iv] = min(emin[iv], MREDUCE(np.min(div_utau_Ds), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(div_utau_Ds), op=MMAX))
                emean[iv]+= MREDUCE(psum(div_utau_Ds), op=MSUM)
                iv+=1

                emin[iv] = min(emin[iv], MREDUCE(np.min(div_utau_Dd), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(div_utau_Dd), op=MMAX))
                emean[iv]+= MREDUCE(psum(div_utau_Dd), op=MSUM)
                iv+=1

                tauS_Ds *= -1.0
                emin[iv] = min(emin[iv], MREDUCE(np.min(tauS_Ds), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(tauS_Ds), op=MMAX))
                emean[iv]+= MREDUCE(psum(tauS_Ds), op=MSUM)
                iv+=1

                tauS_Dd *= -1.0
                emin[iv] = min(emin[iv], MREDUCE(np.min(tauS_Dd), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(tauS_Dd), op=MMAX))
                emean[iv]+= MREDUCE(psum(tauS_Dd), op=MSUM)
                iv+=1

                div_Ku_res *= -1.0
                emin[iv] = min(emin[iv], MREDUCE(np.min(div_Ku_res), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(div_Ku_res), op=MMAX))
                emean[iv]+= MREDUCE(psum(div_Ku_res), op=MSUM)
                iv+=1

                div_reiu_res *= -1.0
                emin[iv] = min(emin[iv], MREDUCE(np.min(div_reiu_res), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(div_reiu_res), op=MMAX))
                emean[iv]+= MREDUCE(psum(div_reiu_res), op=MSUM)
                iv+=1

                emin[iv] = min(emin[iv], MREDUCE(np.min(div_Su_res), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(div_Su_res), op=MMAX))
                emean[iv]+= MREDUCE(psum(div_Su_res), op=MSUM)
                iv+=1

                emin[iv] = min(emin[iv], MREDUCE(np.min(div_Du_res), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(div_Du_res), op=MMAX))
                emean[iv]+= MREDUCE(psum(div_Du_res), op=MSUM)
                iv+=1

                div_Q_res *= -1.0
                emin[iv] = min(emin[iv], MREDUCE(np.min(div_Q_res), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(div_Q_res), op=MMAX))
                emean[iv]+= MREDUCE(psum(div_Q_res), op=MSUM)
                iv+=1

                div_psi_int *= -1.0
                emin[iv] = min(emin[iv], MREDUCE(np.min(div_psi_int), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(div_psi_int), op=MMAX))
                emean[iv]+= MREDUCE(psum(div_psi_int), op=MSUM)
                iv+=1

                div_psi_Q *= -1.0
                emin[iv] = min(emin[iv], MREDUCE(np.min(div_psi_Q), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(div_psi_Q), op=MMAX))
                emean[iv]+= MREDUCE(psum(div_psi_Q), op=MSUM)
                iv+=1

                # repsd_res
                vm = -5./3.*mu_res*dil2_res
                emin[iv] = min(emin[iv], MREDUCE(np.min(vm), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(vm), op=MMAX))
                emean[iv]+= MREDUCE(psum(vm), op=MSUM)
                iv+=1

                # repss_res
                vm = -2.0*mu_res*(S2_res - (1./3.)*dil2_res)
                emin[iv] = min(emin[iv], MREDUCE(np.min(vm), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(vm), op=MMAX))
                emean[iv]+= MREDUCE(psum(vm), op=MSUM)
                iv+=1

                # reps_res
                vm -= 5./3.*mu_res*dil2_res
                emin[iv] = min(emin[iv], MREDUCE(np.min(vm), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(vm), op=MMAX))
                emean[iv]+= MREDUCE(psum(vm), op=MSUM)
                iv+=1

                # Pdil_res
                vm = P_bar*dil_res
                emin[iv] = min(emin[iv], MREDUCE(np.min(vm), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(vm), op=MMAX))
                emean[iv]+= MREDUCE(psum(vm), op=MSUM)
                iv+=1

                # rek_res_res_tran
                vm = div_Su_res + div_Du_res + div_Ku_res + gamma1*div_reiu_res
                emin[iv] = min(emin[iv], MREDUCE(np.min(vm), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(vm), op=MMAX))
                emean[iv]+= MREDUCE(psum(vm), op=MSUM)
                iv+=1

                # rek_res_sgs_tran =
                vm = div_utau_Ds + div_utau_Dd + div_utau_As + div_utau_Ad
                emin[iv] = min(emin[iv], MREDUCE(np.min(vm), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(vm), op=MMAX))
                emean[iv]+= MREDUCE(psum(vm), op=MSUM)
                iv+=1

                # rek_res_sgs_flux =
                vm = tauS_As + tauS_Ad + tauS_Ds + tauS_Dd
                emin[iv] = min(emin[iv], MREDUCE(np.min(vm), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(vm), op=MMAX))
                emean[iv]+= MREDUCE(psum(vm), op=MSUM)
                iv+=1

                # rei_bar_res_tran
                vm = div_reiu_res + div_Q_res
                emin[iv] = min(emin[iv], MREDUCE(np.min(vm), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(vm), op=MMAX))
                emean[iv]+= MREDUCE(psum(vm), op=MSUM)
                iv+=1

                # rei_bar_sgs_tran
                vm = div_psi_int + div_psi_Q
                emin[iv] = min(emin[iv], MREDUCE(np.min(vm), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(vm), op=MMAX))
                emean[iv]+= MREDUCE(psum(vm), op=MSUM)
                iv+=1

                # rei_bar_sgs_flux
                vm = Phi_Ds + Phi_Dd - Phi_P
                emin[iv] = min(emin[iv], MREDUCE(np.min(vm), op=MMIN))
                emax[iv] = max(emax[iv], MREDUCE(np.max(vm), op=MMAX))
                emean[iv]+= MREDUCE(psum(vm), op=MSUM)
                iv+=1

                n = gc.collect()
                if comm.rank % 128 == 0:
                    print(update(timeofday(), Glabel, tstep, run, comm.rank,
                                 'min, max, means computed'))
                # -------------------------------------------------------------

            emean /= Ne
            emavg /= Ne*emean[0]

            for i in xrange(nv):
                tmin[i] = min(tmin[i], emin[i])
                tmax[i] = max(tmax[i], emax[i])
            tmean += emean
            tmavg += emavg

            # -----------------------------------------------------------------

        tmean /= nt
        tmavg /= nt
        rhom = tmean[0]

        # ---------------------------------------------------------------------
        # BEGIN ANALYSIS
        # ---------------------------------------------------------------------

        for it in xrange(its, ite+1, tint):
            tstep = str(it).zfill(4)
            if comm.rank % 128 == 0:
                print(update(timeofday(), Glabel, tstep, comm.rank,
                             'time iteration started for analysis', ''))

            for ir in xrange(irs, ire+1, rint):
                run = str(ir).zfill(2)
                analyzer.prefix = '%s-%s_%s_%s-' % (pid, Glabel, tstep, run)
                analyzer.mpi_moments_file = ('%s%s%s.moments' % (
                        analyzer.odir, analyzer.prefix, 'filtered_transport'))

                if comm.rank == 0:
                    try:
                        os.remove(analyzer.mpi_moments_file)
                    except:
                        pass

                if comm.rank % 128 == 0:
                    print(update(timeofday(), Glabel, tstep, run, comm.rank,
                                 'run iteration started'))

                rho = reader.get_variable(fname(prefix[0], tstep, ir))
                u[0] = reader.get_variable(fname(prefix[1], tstep, ir))
                u[1] = reader.get_variable(fname(prefix[2], tstep, ir))
                u[2] = reader.get_variable(fname(prefix[3], tstep, ir))
                re = reader.get_variable(fname(prefix[4], tstep, ir))

                # -------------------------------------------------------------

                # LES solution fields
                P = re = gamma1*(re - rho*0.5*np.sum(np.square(u), axis=0))

                rho_bar = scalar_filter(rho, Ghat)
                ru_bar = vector_filter(rho*u, Ghat)
                P_bar = scalar_filter(P, Ghat)

                u_tilde = ru_bar = ru_bar/rho_bar

                # Inaccessible filtered fields
                # rek_bar = scalar_filter(rek, Ghat)

                # Derived DNS and LES fields
                # rek = rho*0.5*np.sum(np.square(u), axis=0)
                mu = tcoef*np.power(P/rho/R, texp)

                Sij = analyzer.grad(u)
                Sij = 0.5*(Sij + np.rollaxis(Sij, 1))

                dil = np.einsum('mm...', Sij)
                dil2 = np.square(dil)
                S2 = np.sum(np.square(Sij), axis=(0, 1))

                div_psi_int = analyzer.div(vector_filter(igamma1*P*u, Ghat))
                div_psi_Q = analyzer.div(vector_filter(
                                mu*analyzer.scl_grad(P/rho/R), Ghat))

                Phi_P = scalar_filter(P*dil, Ghat)
                Phi_Ds = scalar_filter(2.0*mu*(S2 - (1./3.)*dil2), Ghat)
                Phi_Dd = scalar_filter(5./3.*mu*dil2, Ghat)

                # # Inconsistently-filtered fields
                # u_bar = vector_filter(u)
                # T_bar = scalar_filter(T, Ghat)
                # dil_bar = scalar_filter(dil, Ghat)
                # dil2_bar= scalar_filter(np.square(dil), Ghat)

                if comm.rank % 128 == 0:
                    print(update(timeofday(), Glabel, tstep, run, comm.rank,
                                 'DNS fields computed and filtered'))
                # -------------------------------------------------------------

                rek_res = 0.5*rho_bar*np.sum(np.square(u_tilde), axis=0)
                # rek_sgs = rek_bar - rek_res
                # rek_tke = rek - rek_bar
                mu_res = tcoef*np.power(P_bar/(rho_bar*R), texp)

                Sij_res = Sij = analyzer.grad(u_tilde)
                Sij_res = 0.5*(Sij_res + np.rollaxis(Sij_res, 1))

                dil_res = np.einsum('mm...', Sij_res)
                dil2_res = dil2 = np.square(dil_res)
                S2_res = S2 = np.sum(np.square(Sij_res), axis=(0, 1))

                # Ss_res= np.sqrt(S2_res - (1./3.)*dil2_res)
                # repss_res = S2_res = 2.0*mu_res*(S2_res - (1./3.)*dil2_res)
                # repsd_res = dil2_res = 5./3.*mu_res*dil2_res

                Phi_P-= P_bar*dil_res
                Phi_Ds-= 2.0*mu_res*(S2_res - (1./3.)*dil2_res)
                Phi_Dd-= 5./3.*mu_res*dil2_res

                div_utau_As = np.zeros_like(rho)
                tauS_As = np.zeros_like(rho)
                div_utau_Ds = np.zeros_like(rho)
                tauS_Ds = np.zeros_like(rho)

                for j in range(3):
                    for i in range(3):
                        tau = scalar_filter(rho*u[i]*u[j], Ghat)
                        tau-= rho_bar*u_tilde[i]*u_tilde[j]

                        div_utau_As+= analyzer.deriv(tau*u_tilde[i], dim=j)
                        tauS_As += tau*Sij_res[j, i]

                        tau = mu*(analyzer.deriv(u[i], dim=j)
                                  + analyzer.deriv(u[j], dim=i))
                        tau = scalar_filter(tau, Ghat)

                        div_utau_Ds+= analyzer.deriv(tau*u_tilde[i], dim=j)
                        tauS_Ds += tau*Sij_res[j, i]

                tau = scalar_filter(rho*np.sum(np.square(u), axis=0), Ghat)
                div_utau_Ad = (1./3.)*analyzer.div(tau*u_tilde)
                div_utau_As-= div_utau_Ad
                tauS_Ad = (1./3.)*tau*dil_res
                tauS_As-= tauS_Ad

                tau = scalar_filter(mu*dil, Ghat)
                div_utau_Dd = analyzer.div(tau*u_tilde)
                div_utau_Ds-= (2./3.)*div_utau_Dd
                div_utau_Dd*= 5./3.
                tauS_Dd = tau*dil_res
                tauS_Ds-= (2./3.)*tauS_Dd
                tauS_Dd*= 5./3.

                div_Ku_res = analyzer.div(rek_res*u_tilde)
                div_reiu_res = analyzer.div(igamma1*P_bar*u_tilde)
                div_Su_res = 2.0*analyzer.div(mu_res*np.sum(
                                        Sij_res*u_tilde.reshape(s), axis=1))
                div_Du_res = analyzer.div(mu_res*dil_res*u_tilde)
                div_Su_res-= (2./3.)*div_Du_res
                div_Q_res = analyzer.div(mu_res*analyzer.scl_grad(
                                                        P_bar/(rho_bar*R)))

                div_psi_int-= div_reiu_res
                div_psi_Q-= div_Q_res

                # clearing out huge chunk of memory
                tau = dil = mu = P = re = None
                Sij_res = Sij = None

                div_utau_As *= -1.0
                div_utau_Ad *= -1.0
                tauS_Ds *= -1.0
                tauS_Dd *= -1.0
                div_Ku_res *= -1.0
                div_reiu_res *= -1.0
                div_Q_res *= -1.0
                div_psi_int *= -1.0
                div_psi_Q *= -1.0

                n = gc.collect()
                if comm.rank % 128 == 0:
                    print(update(timeofday(), Glabel, tstep, run, comm.rank,
                                 'LES fields computed'))
                # -------------------------------------------------------------

                iv = 0

                # rho_bar
                iv+=1

                # dil_res
                scalar_analysis(analyzer, dil_res, (tmin[iv], tmax[iv]),
                                tmavg[iv], rho_bar, rhom, 'dil_res',
                                'resolved dilatation', '\\breve{{\Theta}}')
                iD = iv
                iv+=1

                # dil2_res
                scalar_analysis(analyzer, dil2_res, (tmin[iv], tmax[iv]),
                                tmavg[iv], rho_bar, rhom, 'dil2_res',
                                'resolved squared dilatation',
                                '\\breve{{\Theta}}^2')
                iD2 = iv
                iv+=1

                # S2_res
                scalar_analysis(analyzer, S2_res, (tmin[iv], tmax[iv]),
                                tmavg[iv], rho_bar, rhom, 'S2_res',
                                'resolved squared strain', '|\\breve{{S}}|^2')
                iS2 = iv
                iv+=1

                S_res = np.sqrt(S2_res)
                scalar_analysis(analyzer, S_res, (tmin[iv], tmax[iv]),
                                tmavg[iv], rho_bar, rhom, 'S_res',
                                'resolved strain magnitude', '|\\breve{{S}}|')
                iS = iv
                iv+=1

                S2s_res = S2_res - (1./3.)*dil2_res
                scalar_analysis(analyzer, S2s_res, (tmin[iv], tmax[iv]),
                                tmavg[iv], rho_bar, rhom, 'S2s_res',
                                'resolved squared solenoidal strain',
                                '|\\breve{{S}}^s|^2')
                iS2s = iv
                iv+=1

                Ss_res = np.sqrt(S2s_res)
                scalar_analysis(analyzer, Ss_res, (tmin[iv], tmax[iv]),
                                tmavg[iv], rho_bar, rhom, 'Ss_res',
                                'resolved solenoidal strain magnitude',
                                '|\\breve{{S}}^s|')
                iSs = iv
                iv+=1

                # div_utau_As
                scalar_analysis(analyzer, div_utau_As, (tmin[iv], tmax[iv]),
                                tmean[iv], None, None, 'div_utau_As',
                                'div_utau_As',
                                '-%s(\\tau^{{A,s}}_{ij}\widetilde{{u}}_i)' % ddx)
                iv+=1

                # div_utau_Ad
                scalar_analysis(analyzer, div_utau_Ad, (tmin[iv], tmax[iv]),
                                tmean[iv], None, None, 'div_utau_Ad',
                                'div_utau_Ad',
                                '-%s\left(\\frac{{1}}{{3}}\\tau^{{A}}_{mm}\widetilde{{u}}_i\\right)' % ddx)
                iv+=1

                # tauS_As
                scalar_analysis(analyzer, tauS_As, (tmin[iv], tmax[iv]),
                                tmean[iv], None, None, 'tauS_As',
                                'tauS_As', '\\tau^{{A,s}}_{ij}\\breve{{S}}^s_{ij}')
                iv+=1

                # tauS_Ad
                scalar_analysis(analyzer, tauS_Ad, (tmin[iv], tmax[iv]),
                                tmean[iv], None, None, 'tauS_Ad',
                                'tauS_Ad', '\\frac{{1}}{{3}}\\tau^{{A}}_{mm}\\breve{{\Theta}}')
                iv+=1

                # div_utau_Ds
                scalar_analysis(analyzer, div_utau_Ds, (tmin[iv], tmax[iv]),
                                tmean[iv], None, None, 'div_utau_Ds',
                                'div_utau_Ds',
                                '%s(\\tau^{{D,s}}_{ij}\widetilde{{u}}_i)' % ddx)
                iv+=1

                # div_utau_Dd
                scalar_analysis(analyzer, div_utau_Dd, (tmin[iv], tmax[iv]),
                                tmean[iv], None, None, 'div_utau_Dd',
                                'div_utau_Dd',
                                '%s(\\tau^{{D,d}}_{ij}\widetilde{{u}}_i)' % ddx)
                iv+=1

                # tauS_Ds
                scalar_analysis(analyzer, tauS_Ds, (tmin[iv], tmax[iv]),
                                tmean[iv], None, None, 'tauS_Ds',
                                'tauS_Ds', '-\\tau^{{D,s}}_{ij}\\breve{{S}}^s_{ij}')
                iv+=1

                # tauS_Dd
                scalar_analysis(analyzer, tauS_Dd, (tmin[iv], tmax[iv]),
                                tmean[iv], None, None, 'tauS_Dd',
                                'tauS_Dd', '-\\tau^{{D,d}}_{ij}\\breve{{S}}^d_{ij}')
                iv+=1

                # div_Ku_res
                scalar_analysis(analyzer, div_Ku_res, (tmin[iv], tmax[iv]),
                                tmean[iv], None, None, 'div_Ku_res',
                                'div_Ku_res',
                                '-%s(\\breve{{K}}\widetilde{{u}}_j)' % ddx)
                iv+=1

                # div_reiu_res
                scalar_analysis(analyzer, div_reiu_res, (tmin[iv], tmax[iv]),
                                tmean[iv], None, None, 'div_reiu_res',
                                'div_reiu_res',
                                '-%s(\overline{{\\rho e_{{int}}}}\widetilde{{u}}_j)' % ddx)
                iv+=1

                # div_Su_res
                scalar_analysis(analyzer, div_Su_res, (tmin[iv], tmax[iv]),
                                tmean[iv], None, None, 'div_Su_res',
                                'div_Su_res',
                                '%s(\\breve{{\sigma}}^s_{ij}\widetilde{{u}}_i)' % ddx)
                iv+=1

                # div_Du_res
                scalar_analysis(analyzer, div_Du_res, (tmin[iv], tmax[iv]),
                                tmean[iv], None, None, 'div_Du_res',
                                'div_Du_res',
                                '%s(\\breve{{\sigma}}^d_{ij}\widetilde{{u}}_i)' % ddx)
                iv+=1

                # div_Q_res
                scalar_analysis(analyzer, div_Q_res, (tmin[iv], tmax[iv]),
                                tmean[iv], None, None, 'div_Q_res',
                                'div_Q_res',
                                '-%s\left(\\breve{{k}}\\frac{{\partial\widetilde{{T}}}}{{\partial x_j}}\\right)' % ddx)
                iv+=1

                # div_psi_int
                scalar_analysis(analyzer, div_psi_int, (tmin[iv], tmax[iv]),
                                tmean[iv], None, None, 'div_psi_int',
                                'div_psi_int',
                                '-\\frac{{\partial(\psi^{{int}}_j)}}{{\partial x_j}}')
                iv+=1

                div_psi_Q
                scalar_analysis(analyzer, div_psi_Q, (tmin[iv], tmax[iv]),
                                tmean[iv], None, None, 'div_psi_Q',
                                'div_psi_Q',
                                '-\\frac{{\partial(\psi^{{Q}}_j)}}{{\partial x_j}}')
                iv+=1

                # repsd_res
                vm = -5./3.*mu_res*dil2_res
                scalar_analysis(analyzer, vm, (tmin[iv], tmax[iv]), tmean[iv],
                                None, None, 'repsd_res', 'repsd_res',
                                '-\\frac{{5}}{{3}}\\breve{{\mu}}\\breve{{\Theta}}^2')
                iv+=1

                # repss_res
                vm = -2.0*mu_res*S2s_res
                scalar_analysis(analyzer, vm, (tmin[iv], tmax[iv]), tmean[iv],
                                None, None, 'repss_res', 'repss_res',
                                '-2\\breve{{\mu}}|\\breve{{S}}^s|^2')
                iv+=1

                # reps_res
                vm -= 5./3.*mu_res*dil2_res
                scalar_analysis(analyzer, vm, (tmin[iv], tmax[iv]), tmean[iv],
                                None, None, 'reps_res', 'reps_res',
                                '\\breve{{\sigma}}_{ij}\\breve{{S}}_{ij}')
                iv+=1

                # Pdil_res
                vm = P_bar*dil_res
                scalar_analysis(analyzer, vm, (tmin[iv], tmax[iv]), tmean[iv],
                                None, None, 'Pdil_res', 'Pdil_res',
                                '\overline{{P}}\\breve{{\Theta}}')
                iv+=1

                # rek_res_res_tran
                vm = div_Su_res + div_Du_res + div_Ku_res + gamma1*div_reiu_res
                scalar_analysis(analyzer, vm, (tmin[iv], tmax[iv]), tmean[iv],
                                None, None,
                                'rek_res_res_tran',
                                'rek_res_res_tran',
                                '$\\breve{{K}}$ total resolved transport')
                iv+=1

                # rek_res_sgs_tran
                vm = div_utau_Ds + div_utau_Dd + div_utau_As + div_utau_Ad
                scalar_analysis(analyzer, vm, (tmin[iv], tmax[iv]), tmean[iv],
                                None, None,
                                'rek_res_sgs_tran',
                                'rek_res_sgs_tran',
                                '$\\breve{{K}}$ total SGS transport')
                iv+=1

                # rek_res_sgs_flux
                vm = tauS_As + tauS_Ad + tauS_Ds + tauS_Dd
                scalar_analysis(analyzer, vm, (tmin[iv], tmax[iv]), tmean[iv],
                                None, None,
                                'rek_res_sgs_flux',
                                'rek_res_sgs_flux',
                                '$\\breve{{K}}$ total SGS flux')
                iv+=1

                # rei_bar_res_tran
                vm = div_reiu_res + div_Q_res
                scalar_analysis(analyzer, vm, (tmin[iv], tmax[iv]), tmean[iv],
                                None, None,
                                'rei_bar_res_tran',
                                'rei_bar_res_tran',
                                '$\overline{{\\rho e_{{int}}}}$ total resolved transport')
                iv+=1

                # rei_bar_sgs_tran
                vm = div_psi_int + div_psi_Q
                scalar_analysis(analyzer, vm, (tmin[iv], tmax[iv]), tmean[iv],
                                None, None,
                                'rei_bar_sgs_tran',
                                'rei_bar_sgs_tran',
                                '$\overline{{\\rho e_{{int}}}}$ total SGS transport')
                iv+=1

                # rei_bar_cf_flux
                vm = Phi_Ds + Phi_Dd - Phi_P
                scalar_analysis(analyzer, vm, (tmin[iv], tmax[iv]), tmean[iv],
                                None, None,
                                'rei_bar_cf_flux',
                                'rei_bar_cf_flux',
                                '$\overline{{\\rho e_{{int}}}}$ total cross-filter flux')
                iv+=1

                n = gc.collect()
                if comm.rank % 128 == 0:
                    print(update(timeofday(), Glabel, tstep, run, comm.rank,
                                 'statistics computed'))
                # -------------------------------------------------------------

    if comm.rank == 0:
        print("\nPython MPI job `filtered_transport_analysis' finished at {}"
              .format(timeofday()))

    return

###############################################################################
if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: .8e}'.format})
    filtered_transport_analysis(get_inputs())
