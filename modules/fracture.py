# -*- coding: utf-8 -*-

from collections import defaultdict
from numpy import pi as PI
from numpy import zeros
from numpy import ones
from numpy import sin
from numpy import cos
from numpy import arange
from numpy import column_stack
from numpy import row_stack
from numpy.random import random

from numpy import float32 as npfloat
from numpy import int32 as npint

TWOPI = PI*2
HPI = PI*0.5


class Fracture(object):
  def __init__(
      self,
      frac_dot,
      frac_dst,
      frac_stp,
      initial_sources,
      frac_spd=1.0,
      frac_diminish=1.0,
      frac_spawn_diminish=1.0,
      threads = 256,
      zone_leap = 200,
      nmax = 100000
      ):
    self.itt = 0

    self.frac_dot = frac_dot
    self.frac_dst = frac_dst
    self.frac_stp = frac_stp
    self.frac_spd = frac_spd

    self.frac_diminish = frac_diminish
    self.spawn_diminish = frac_spawn_diminish

    self.threads = threads
    self.zone_leap = zone_leap

    self.nmax = nmax

    self.num = 0
    self.fnum = 0
    self.anum = 0

    self.__init(initial_sources)
    self.__cuda_init()

  def __init(self, initial_sources):
    num = len(initial_sources)
    self.num = num

    nz = int(0.5/self.frac_dst)
    self.nz = nz
    self.nz2 = nz**2
    nmax = self.nmax

    self.xy = zeros((nmax, 2), npfloat)
    self.xy[:num,:] = initial_sources[:,:]

    self.fid_node = zeros((nmax, 2), npint)
    self.fid_node[:,:] = -1

    self.visited = zeros((nmax, 1), npint)
    self.visited[:, :] = -1
    self.active = zeros((nmax, 1), npint)
    self.active[:, :] = -1

    self.diminish = ones((nmax, 1), npfloat)
    self.spd = ones((nmax, 1), npfloat)
    self.dxy = ones((nmax, 2), npfloat)
    self.ndxy = ones((nmax, 2), npfloat)

    self.zone_num = zeros(self.nz2, npint)
    self.zone_node = zeros(self.nz2*self.zone_leap, npint)

  def __cuda_init(self):
    import pycuda.autoinit
    from .helpers import load_kernel

    self.cuda_agg = load_kernel(
        'modules/cuda/agg.cu',
        'agg',
        subs={'_THREADS_': self.threads}
        )
    self.cuda_step = load_kernel(
        'modules/cuda/calc_stp.cu',
        'calc_stp',
        subs={
          '_THREADS_': self.threads,
          '_PROX_': self.zone_leap
          }
        )

  def get_nodes(self):
    return self.xy[:self.num, :]

  def get_fractures(self):
    res = defaultdict(list)

    for fid, node in self.fid_node[:self.fnum, :]:
      res[fid].append(self.xy[node, :])

    return [row_stack(v) for k, v in res.items()]

  def get_fractures_inds(self):
    res = defaultdict(list)

    for fid, node in self.fid_node[:self.fnum, :]:
      res[fid].append(node)

    return [v for k, v in res.items()]

  def blow(self, n, xy):
    a = random(size=n)*TWOPI
    dxy = column_stack((
        cos(a),
        sin(a)
        ))

    num = self.num
    fnum = self.fnum
    anum = self.anum

    self.xy[num, :] = xy

    self.dxy[fnum:fnum+n, :] = dxy
    self.spd[fnum:fnum+n, :] = self.frac_spd

    new = arange(fnum, fnum+n)

    fid_node = column_stack((
        new,
        ones(n, npint)*num
        ))

    self.fid_node[new, :] = fid_node
    self.visited[num] = 1

    self.active[anum:anum+n, 0] = new[:]

    self.num += 1
    self.anum += n
    self.fnum += n

  def _do_steps(self, active, ndxy):
    num = self.num
    fnum = self.fnum
    # anum = self.anum

    mask = ndxy[:, 0] >= -1.0
    n = mask.sum()
    if n<1:
      return False

    ndxy = ndxy[mask, :]

    new = arange(fnum, fnum+n)
    self.dxy[new, :] = ndxy
    self.xy[num:num+n, :] = self.xy[self.fid_node[active[mask, 0], 1].squeeze(), :] + \
        ndxy*self.frac_stp
    self.spd[new, :] = self.frac_spd

    fid_node = column_stack((
        self.fid_node[active[mask], 0],
        # new[:]
        arange(num, num+n)
        ))
    self.fid_node[new, :] = fid_node

    self.visited[num:num+n] = 1

    self.active[mask.nonzero()[0], 0] = new

    self.num += 1
    # self.anum += n
    self.fnum += n

    # print('ndxy\n', ndxy, '\n')
    return True

  def step(self):
    import pycuda.driver as drv

    self.itt += 1

    num = self.num
    fnum = self.fnum
    anum = self.anum

    xy = self.xy[:num, :]
    active = self.active[:anum]
    fid_node = self.fid_node[:fnum]
    dxy = self.dxy[:fnum, :]
    ndxy = self.ndxy[:anum, :]

    self.zone_num[:] = 0

    self.cuda_agg(
        npint(self.nz),
        npint(self.zone_leap),
        npint(num),
        drv.In(xy),
        drv.InOut(self.zone_num),
        drv.InOut(self.zone_node),
        block=(self.threads,1,1),
        grid=(num//self.threads + 1, 1)
        )

    ndxy[:,:] = -10

    self.cuda_step(
        npint(self.nz),
        npint(self.zone_leap),
        npint(num),
        npint(fnum),
        npint(anum),
        npfloat(self.frac_dot),
        npfloat(self.frac_dst),
        npfloat(self.frac_stp),
        drv.In(fid_node),
        drv.In(active),
        drv.In(xy),
        drv.In(dxy),
        drv.Out(ndxy),
        drv.In(self.zone_num),
        drv.In(self.zone_node),
        block=(self.threads,1,1),
        grid=(anum//self.threads + 1, 1)
        )

    print('active\n', active, '\n')
    print('fid_node\n', fid_node, '\n')
    print('ndxy\n', ndxy, '\n')
    res = self._do_steps(active, ndxy)

    return res


