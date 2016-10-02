#!/usr/bin/python3
# -*- coding: utf-8 -*-


BACK = [1, 1, 1, 1]
FRONT = [0, 0, 0, 0.01]

NMAX = 10**7
SIZE = 3000
ONE = 1./SIZE
LINEWIDTH = ONE*1.5

FRAC_DOT = 0.8
FRAC_DST = 70*ONE
FRAC_STP = ONE
FRAC_SPD = 1.0

FRAC_DIMINISH = 0.997
FRAC_SPAWN_DIMINISH = 0.9

SPAWN_ANGLE = 0.2
SPAWN_FACTOR = 0.09

THREADS = 1024
ZONE_LEAP = 1024*10

EDGE = 0.1
SOURCES = 10000

DRAW_ITT = 100

DBG = False

CMULT = 20
GRAINS = 30


def show(sand, f):
  from numpy import ones

  sand.set_bg(BACK)

  for frac in f.get_fractures():
    frac = frac.astype('double')
    a = frac[1:, :]
    b = frac[:-1:, :]
    sand.paint_filled_circle_strokes(
        a, b,
        ones(len(a), 'float')*FRAC_STP*0.9, CMULT,
        ones(len(a), 'int')*GRAINS
        )


def main():
  from modules.fracture import Fracture
  from numpy.random import random
  from iutils.random import darts_rect
  from time import time
  from sand import Sand

  sand = Sand(SIZE)
  sand.set_bg(BACK)
  sand.set_rgba(FRONT)

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

  for _ in range(40):
    F.blow(1, EDGE+random((1, 2))*(1.0-2.0*EDGE))

  while True:

    res = F.step()

    F.frac_front(factor=SPAWN_FACTOR, angle=SPAWN_ANGLE, dbg=DBG)

    if not F.itt % DRAW_ITT or not res:
      print('itt', F.itt, 'num', F.num, 'fnum', F.fnum, 'anum', F.anum, 'time', time()-start)
      show(sand, F)
      name = fn.name()+'.png'
      sand.write_to_png(name)

    if not res:
      print('done')
      return


if __name__ == '__main__':

  main()

