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

FRAC_DOT = 0.95
FRAC_DST = 0.03
FRAC_STP = ONE
FRAC_SPD = 1.0

FRAC_DIMINISH = 0.997
FRAC_SPAWN_DIMINISH = 0.9

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

  # nodes = f.get_nodes()
  # render.set_front(FRONT)
  # for x, y in nodes:
  #   render.circle(x, y, ONE, fill=False)

  render.set_front(FRONT)
  fractures = f.get_fractures()

  for frac in fractures:
    render.path(frac)


def main():
  from modules.fracture import Fracture
  from iutils.render import Animate
  from numpy.random import random
  from iutils.random import darts_rect
  from time import time

  start = time()

  from fn import Fn
  fn = Fn(prefix='./res/')

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
      frac_spd=FRAC_SPD,
      frac_diminish=FRAC_DIMINISH,
      frac_spawn_diminish=FRAC_SPAWN_DIMINISH,
      zone_leap=ZONE_LEAP,
      nmax=NMAX
      )

  for _ in range(5):
    F.blow(1, EDGE+random((1, 2))*(1.0-2.0*EDGE))

  def wrap(render):
    print('itt', F.itt, 'num', F.num, 'fnum', F.fnum, 'anum', F.anum, 'time', time()-start)
    res = F.step()

    n = F.frac_front(factor=SPAWN_FACTOR, angle=SPAWN_ANGLE, dbg=DBG)
    if n > 0:
      print('new fracs: {:d}'.format(n))

    if not F.itt % DRAW_ITT:
      show(render, F)
      name = fn.name()+'.png'
      render.write_to_png(name)

    return res

  render = Animate(SIZE, BACK, FRONT, wrap)
  render.set_line_width(LINEWIDTH)
  render.start()


if __name__ == '__main__':

  main()

