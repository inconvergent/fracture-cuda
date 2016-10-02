#!/usr/bin/python3
# -*- coding: utf-8 -*-


BACK = [1, 1, 1, 1]
FRONT = [0, 0, 0, 0.2]
LIGHT = [0, 0, 0, 0.2]
CYAN = [0, 0.5, 0.5, 0.4]
BLUE = [0, 0, 1, 0.3]

NMAX = 10**6
SIZE = 1000
ONE = 1./SIZE
LINEWIDTH = ONE*1.1

FRAC_DOT = 0.99
FRAC_DST = 0.1
FRAC_STP = ONE*4
FRAC_SPD = 1.0

FRAC_DIMINISH = 0.997
FRAC_SPAWN_DIMINISH = 0.9

SPAWN_ANGLE = 0.0
SPAWN_FACTOR = 0.03

THREADS = 512
ZONE_LEAP = 1024*10

EDGE = 0.1
SOURCES = 20000


def show(render, f):
  render.clear_canvas()

  nodes = f.get_nodes()
  fractures = f.get_fractures()

  render.set_front(FRONT)
  for x, y in nodes:
    render.circle(x, y, ONE, fill=False)

  render.set_front(CYAN)
  for frac in fractures:
    for x, y in frac:
      render.circle(x, y, 0.5*FRAC_STP, fill=True)
    render.circle(x, y, FRAC_DST, fill=False)

  # print('show fracs')
  # for frac in f.get_fractures_inds():
  #   print(frac)


def main():
  from modules.fracture import Fracture
  from iutils.render import Animate
  from numpy.random import random
  from iutils.random import darts_rect

  # from fn import Fn
  # fn = Fn(prefix='./res/')

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

  for _ in range(1):
    F.blow(1, EDGE+random((1, 2))*(1.0-2.0*EDGE))

  def wrap(render):
    print('itt', F.itt, 'num', F.num, 'fnum', F.fnum, 'anum', F.anum)
    res = F.step()

    if not F.itt % 1:
      show(render, F)
      # name = fn.name()+'.png'
      # render.write_to_png(name)

    n = F.frac_front(factor=SPAWN_FACTOR, angle=SPAWN_ANGLE)
    print('new fracs: {:d}'.format(n))
    return res

  render = Animate(SIZE, BACK, FRONT, wrap)
  render.set_line_width(ONE)
  render.start()


if __name__ == '__main__':

  main()

