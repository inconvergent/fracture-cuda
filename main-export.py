#!/usr/bin/python3
# -*- coding: utf-8 -*-


BACK = [1, 1, 1, 1]
FRONT = [0, 0, 0, 0.8]
LIGHT = [0, 0, 0, 0.2]
CYAN = [0, 0.5, 0.5, 0.4]
BLUE = [0, 0, 1, 0.3]

NMAX = 10**6
SIZE = 1000
ONE = 1./SIZE
LINEWIDTH = ONE*1.5

FRAC_DOT = 0.8
FRAC_DST = 30*ONE
FRAC_STP = ONE

SPAWN_ANGLE = 0.0
SPAWN_FACTOR = 0.04

THREADS = 512
ZONE_LEAP = 1024*10

EDGE = 0.1
SOURCES = 30000

DRAW_ITT = 20

DBG = False


def show(render, f):
  render.clear_canvas()

  render.set_front(FRONT)
  fractures = f.get_fractures()

  for frac in fractures:
    render.path(frac)

def export(name, f):
  from numpy import array
  from iutils.ioOBJ import export_2d as export

  vertices = f.xy[:f.num, :]
  lines = [array(l, 'int') for l in f.get_fractures_inds()]

  print(name)
  export('fractures', name, vertices, lines=lines)


def main():
  from modules.fracture import Fracture
  from iutils.render import Animate
  from numpy.random import random
  from iutils.random import darts_rect
  from time import time
  from fn import Fn

  fn = Fn(prefix='./res/')

  start = time()

  initial_sources = darts_rect(
      SOURCES,
      0.5, 0.5,
      1.0-2.0*EDGE, 1.0-2.0*EDGE,
      FRAC_STP
      )

  F = Fracture(
      FRAC_DOT,
      FRAC_DST,
      FRAC_STP,
      initial_sources=initial_sources,
      zone_leap=ZONE_LEAP,
      nmax=NMAX
      )

  for _ in range(20):
    F.blow(1, EDGE+random((1, 2))*(1.0-2.0*EDGE))

  def wrap(render):
    print('itt', F.itt, 'num', F.num, 'fnum', F.fnum, 'anum', F.anum, 'time', time()-start)
    res = F.step()

    n = F.frac_front(factor=SPAWN_FACTOR, angle=SPAWN_ANGLE, dbg=DBG)
    if n > 0:
      print('new fracs: {:d}'.format(n))

    if not F.itt % DRAW_ITT:
      show(render, F)

    return res

  render = Animate(SIZE, BACK, FRONT, wrap)
  render.set_line_width(LINEWIDTH)
  render.start()

  name = fn.name()
  render.write_to_png(name+'.png')
  export(name+'.2obj', F)


if __name__ == '__main__':

  main()

