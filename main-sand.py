#!/usr/bin/python3
# -*- coding: utf-8 -*-

from numpy.random import randint
from numpy.random import random


BACK = [1, 1, 1, 1]
FRONT = [0, 0, 0, 0.01]

NMAX = 10**8
SIZE = 3000
ONE = 1./SIZE

FRAC_DOT = 0.99 + random() * 0.02
FRAC_DST = 30*ONE
FRAC_STP = ONE

SPAWN_ANGLE = 0.0
SPAWN_FACTOR = random() * 0.01 + 0.012

THREADS = 1024
ZONE_LEAP = 1024*20

EDGE = 0.05
SOURCES = 500000

INIT_FRACS = randint(20, 35)

DRAW_ITT = 400

DBG = False

CMULT = 20
GRAINS = 35

FRAC_RAD = 2.5*ONE


def show(sand, f):
  from numpy import ones

  sand.set_bg(BACK)

  for frac in f.get_fractures():
    frac = frac.astype('double')
    a = frac[1:, :]
    b = frac[:-1:, :]
    sand.paint_filled_circle_strokes(
        a, b,
        ones(len(a), 'float')*FRAC_RAD, CMULT,
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
      zone_leap=ZONE_LEAP,
      nmax=NMAX
      )

  for _ in range(INIT_FRACS):
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
      break


if __name__ == '__main__':
  main()

