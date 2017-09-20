"""
"""

from mpi4py import MPI
import numpy as np
from math import *
import time
import sys
import getopt

world_comm = MPI.COMM_WORLD
help_string = ("Athena analyzer command line options:\n"
               "-i <input directory>,\t defualt: 'data/'\n"
               "-o <output directory>,\t defualt: 'analysis/'\n"
               "-p <problem ID>,\t\t defualt: 'no_problem_id'\n"
               "-N <Nx>,\t\t default: 512\n"
               "-g <gamma>,\t\t default: 1.4\n"
               "-L <L>,\t\t\t default: 1.0\n"
               "-r <irs:ire:rint>,\t\t default: 1:20:1\n"
               "-t <its:ite:tint>,\t\t default: 1:20:1\n"
               "-R <R>,\t\t\t default: 8.3144598e7/21\n"
               "--Texp <texp>,\t\t default: 0.7\n"
               "--Tcoef <tcoef>,\t default: 3.1e-6\n"
               "--Tmp0 <tmp0>,\t\t default: 293.0\n")


def timeofday():
    return time.strftime("%H:%M:%S")


def get_inputs():
    """
    Command Line Options:
    ---------------------
    -i <input directory>
    -o <output directory>
    -p <problem ID>
    -L <Domain box size>
    -N <grid cells per dimension>
    -c <CFL coefficient>
    --tlimit <simulation time limit>
    --dt_rst
    --dt_bin
    --dt_hst
    --dt_spec
    --nu <kinematic viscosity>
    --eps <energy injection rate>
    --Uinit <Urms of random IC>
    --k_exp <power law exponent of random IC>
    --k_peak <exponential decay scaling>
    """

    # import 'defaults' from parameters file
    from HIT_parameters import idir, odir, pid, L, N, cfl, tlimit, dt_rst, \
        dt_bin, dt_hst, dt_spec, nu, eps_inj, Urms, k_exp, k_peak

    help_string = ("spectralLES HIT solver command line options:\n"
                   "-i <input directory>\n"
                   "-o <output directory>\n"
                   "-p <problem ID>\n"
                   "-L <Domain box size>\n"
                   "-N <grid cells per dimension>\n"
                   "-c <CFL coefficient>\n"
                   "--tlimit <simulation time limit>\n"
                   "--dt_rst <output rate of restart fields>\n"
                   "--dt_bin <output rate of single-precision fields>\n"
                   "--dt_hst <output rate of analysis histograms>\n"
                   "--dt_spec <output rate of 1D velocity spectra>\n"
                   "--nu <kinematic viscosity>\n"
                   "--eps <energy injection rate>\n"
                   "--Uinit <Urms of random IC>\n"
                   "--k_exp <power law exponent of random IC>\n"
                   "--k_peak <exponential decay scaling>\n")

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:p:L:N:c:",
                                   ["tlimit=", "dt_rst=", "dt_bin=", "dt_hst=",
                                    "dt_spec=", "nu=", "eps=", "Uinit=",
                                    "k_exp=", "k_peak="])
    except getopt.GetoptError as e:
        if comm.rank == 0:
            print(e)
            print(help_string)
        MPI.Finalize()
        sys.exit(999)
    except Exception as e:
        if comm.rank == 0:
            print('Unknown exception while getting input arguments!')
            print(e)
        MPI.Finalize()
        try:
            sys.exit(e.errno)
        except:
            sys.exit(999)

    for opt, arg in opts:
        try:
            if opt=='-h':
                if comm.rank == 0:
                    print(help_string)
                MPI.Finalize()
                sys.exit(1)
            elif opt=='-i':
                idir = arg
                if comm.rank == 0:
                    print('input directory:\t'+idir)
            elif opt=='-o':
                odir = arg
                if comm.rank == 0:
                    print('output directory:\t'+odir)
            elif opt=='-p':
                pid = arg
                if comm.rank == 0:
                    print('problem ID:\t\t'+pid)
            elif opt=='-L':
                L = float(arg)
                if comm.rank == 0:
                    print('L:\t\t\t{}'.format(L))
            elif opt=='-N':
                N = int(arg)
                if comm.rank == 0:
                    print('N:\t\t\t{}'.format(N))
            elif opt=='-c':
                cfl = float(arg)
                if comm.rank == 0:
                    print('CFL:\t\t\t{}'.format(cfl))
            elif opt=='--tlimit':
                tlimit = float(arg)
                if comm.rank == 0:
                    print('tlimit:\t\t\t{}'.format(tlimit))
            elif opt=='--dt_rst':
                dt_rst = float(arg)
                if comm.rank == 0:
                    print('dt_rst:\t\t\t{}'.format(dt_rst))
            elif opt=='--dt_bin':
                dt_bin = float(arg)
                if comm.rank == 0:
                    print('dt_bin:\t\t\t{}'.format(dt_bin))
            elif opt=='--dt_hst':
                dt_hst = float(arg)
                if comm.rank == 0:
                    print('dt_hst:\t\t\t{}'.format(dt_hst))
            elif opt=='--dt_spec':
                dt_spec = float(arg)
                if comm.rank == 0:
                    print('dt_spec:\t\t\t{}'.format(dt_spec))
            elif opt=='--nu':
                nu = float(arg)
                if comm.rank == 0:
                    print('nu:\t\t\t{}'.format(nu))
            elif opt=='--eps':
                eps_inj = float(arg)
                if comm.rank == 0:
                    print('eps_inj:\t\t\t{}'.format(eps_inj))
            elif opt=='--Uinit':
                Urms = float(arg)
                if comm.rank == 0:
                    print('Uinit:\t\t\t{}'.format(Urms))
            elif opt=='--k_exp':
                k_exp = float(arg)
                if comm.rank == 0:
                    print('k_exp:\t\t\t{}'.format(k_exp))
            elif opt=='--k_peak':
                k_peak = float(arg)
                if comm.rank == 0:
                    print('k_peak:\t\t\t{}'.format(k_peak))
            else:
                if comm.rank == 0:
                    print(help_string)
                MPI.Finalize()
                sys.exit(1)
        except Exception as e:
            if comm.rank == 0:
                print('Unknown exception while reading argument {} '
                      'from option {}!'.format(opt, arg))
                print(e)
            MPI.Finalize()
            sys.exit(e.errno)

    return (idir, odir, pid, L, N, cfl, tlimit, dt_rst, dt_bin, dt_hst,
            dt_spec, nu, eps_inj, Urms, k_exp, k_peak)


def get_inputs():
    """
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
    """
    idir = 'data/'                  # input folder
    odir = 'analysis/'              # input folder
    N = 512                         # linear grid size
    pid = 'no_problem_id'           # problem ID
    L = 1.0                         # domain size
    irs = 1                         # ensemble run index start
    ire = 20                        # ensemble run index end
    rint = 1                        # ensemble run index interval
    its = 1                         # vtk time index start
    ite = 20                        # vtk time index end
    tint = 1                        # vtk time index interval
    gamma = 1.4                     # heat capacity ratio
    R = 8.3144598e7/21              # gas constant
    texp = 0.7                      # transport properties exponent "n"
    tcoef = 3.1e-6                  # transport properties Th1^n coefficient
    tmp0 = 293.0                    # reference temperature

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:p:N:n:M:g:L:r:t:R:",
                                   ["Texp=", "Tcoef=", "Tmp0="])
    except getopt.GetoptError as e:
        if world_comm.rank == 0:
            print e
            print help_string
        MPI.Finalize()
        sys.exit(999)
    except Exception as e:
        if world_comm.rank == 0:
            print ('Unknown exception while getting input arguments!')
            print e
        MPI.Finalize()
        try:
            sys.exit(e.errno)
        except:
            sys.exit(999)

    for opt, arg in opts:
        try:
            if opt=='-h':
                if world_comm.rank == 0:
                    print help_string
                MPI.Finalize()
                sys.exit(1)
            elif opt=='-i':
                idir = arg
                if world_comm.rank == 0:
                    print 'input directory:\t'+idir
            elif opt=='-o':
                odir = arg
                if world_comm.rank == 0:
                    print 'output directory:\t'+odir
            elif opt=='-p':
                pid = arg
                if world_comm.rank == 0:
                    print 'problem ID:\t\t'+pid
            elif opt=='-N':
                N = int(arg)
                if world_comm.rank == 0:
                    print 'N:\t\t\t{}'.format(N)
            elif opt=='-g':
                gamma = float(arg)
                if world_comm.rank == 0:
                    print 'gamma:\t\t\t{}'.format(gamma)
            elif opt=='-L':
                L = float(arg)
                if world_comm.rank == 0:
                    print 'L:\t\t\t{}'.format(L)
            elif opt=='-r':
                try:
                    [irs, ire, rint] = [int(i) for i in arg.split(':')]
                except ValueError as e:
                    if world_comm.rank == 0:
                        print ('Input Error: option -r <irs:ire:rint> requires'
                               ' three integer values separated by colons.')
                        print e
                        print help_string
                    MPI.Finalize()
                    sys.exit(e.errno)
                if world_comm.rank == 0:
                    print 'ensemble runs:\t\t{}'.format((irs, ire, rint))
            elif opt=='-t':
                try:
                    [its, ite, tint] = [int(i) for i in arg.split(':')]
                except ValueError as e:
                    if world_comm.rank == 0:
                        print ('Input Error: option -t <its:ite:tint> requires'
                               ' three integer values separated by colons.')
                        print e
                        print help_string
                    MPI.Finalize()
                    sys.exit(e.errno)
                if world_comm.rank == 0:
                    print 'time steps:\t\t{}'.format((its, ite, tint))
            elif opt=='-R':
                R = float(arg)
                if world_comm.rank == 0:
                    print 'R:\t\t\t{}'.format(R)
            elif opt=='--Texp':
                texp = float(arg)
                if world_comm.rank == 0:
                    print 'texp:\t\t\t{}'.format(texp)
            elif opt=='--Tcoef':
                tcoef = float(arg)
                if world_comm.rank == 0:
                    print 'tcoef:\t\t\t{}'.format(tcoef)
            elif opt=='--Tmp0':
                tmp0 = float(arg)
                if world_comm.rank == 0:
                    print 'tcoef:\t\t\t{}'.format(tmp0)
            else:
                if world_comm.rank == 0:
                    print help_string
                MPI.Finalize()
                sys.exit(1)
        except Exception as e:
            if world_comm.rank == 0:
                print ('Unknown exception while reading argument {} '
                       'from option {}!'.format(opt, arg))
                print e
            MPI.Finalize()
            sys.exit(e.errno)

    args = (idir, odir, pid, N, L, irs, ire, rint, its, ite, tint, gamma, R,
            texp, tcoef, tmp0)

    return args
