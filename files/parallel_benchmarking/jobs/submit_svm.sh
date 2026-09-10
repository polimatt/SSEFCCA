#!/bin/bash

for entry in ./*svm*.job
do
  qsub "$entry"
done
