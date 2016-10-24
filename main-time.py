#!/usr/bin/python3
# -*- coding: utf-8 -*-

BACK = [1, 1, 1, 1]
FRONT = [0, 0, 0, 0.01]


NMAX = 10**8
SIZE = 2000
ONE = 1./SIZE

FRAC_DOT = 0.95
FRAC_DST = 30*ONE
FRAC_STP = ONE
FRAC_SPD = 1.0

FRAC_DIMINISH = 0.997
FRAC_SPAWN_DIMINISH = 0.9

SPAWN_ANGLE = 0.0
SPAWN_FACTOR = 0.04

THREADS = 1024
ZONE_LEAP = 1024*20

EDGE = 0.05
SOURCES = 500000

INIT_FRACS = 20

ITT = 200

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
        ones(len(a), 'float')*FRAC_STP, CMULT,
        ones(len(a), 'int')*GRAINS
        )


def main():
  from modules.fracture import Fracture
  from numpy.random import random
  from numpy import linspace
  from iutils.random import darts_rect
  from time import time

  from numpy.random import seed

  from sand import Sand

  from fn import Fn
  fn = Fn(prefix='./res/')


  sand = Sand(SIZE)
  sand.set_bg(BACK)
  sand.set_rgba(FRONT)

  seed(1)

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
      frac_spd=FRAC_SPD,
      frac_diminish=FRAC_DIMINISH,
      frac_spawn_diminish=FRAC_SPAWN_DIMINISH,
      zone_leap=ZONE_LEAP,
      nmax=NMAX
      )

  for _ in range(INIT_FRACS):
    F.blow(1, EDGE+random((1, 2))*(1.0-2.0*EDGE))

  while True:
    res = F.step()
    F.frac_front(factor=SPAWN_FACTOR, angle=SPAWN_ANGLE, dbg=DBG)

    if not F.itt % ITT or not res:
      print('itt', F.itt, 'num', F.num, 'fnum', F.fnum, 'anum', F.anum, 'time', time()-start)

    if not res:
      print('done')
      break

  show(sand, F)
  name = fn.name()+'.png'
  print(name)
  sand.write_to_png(name)


if __name__ == '__main__':

  main()

