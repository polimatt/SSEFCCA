#!/bin/bash

for entry in ./*.job
do
  qsub "$entry"
done
