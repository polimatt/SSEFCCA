#!/bin/bash

for entry in ./*_accuracy.job
do
  qsub "$entry"
done
