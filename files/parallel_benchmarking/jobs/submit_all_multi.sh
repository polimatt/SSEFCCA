#!/bin/bash

for entry in ./*multi*.job
do
  qsub "$entry"
done
