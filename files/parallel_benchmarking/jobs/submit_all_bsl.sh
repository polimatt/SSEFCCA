#!/bin/bash

for entry in ./*_bsl.job
do
  qsub "$entry"
done
